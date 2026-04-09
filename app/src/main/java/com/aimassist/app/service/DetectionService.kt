package com.aimassist.app.service

import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.graphics.PixelFormat
import android.hardware.HardwareBuffer
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.Image
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Binder
import android.os.Build
import android.os.Handler
import android.os.HandlerThread
import android.os.IBinder
import android.util.DisplayMetrics
import android.util.Log
import android.view.WindowManager
import androidx.core.app.NotificationCompat
import com.aimassist.app.App
import com.aimassist.app.MainActivity
import com.aimassist.app.R
import com.aimassist.app.data.DetectionResult
import com.aimassist.app.data.InferenceStats
import com.aimassist.app.ncnn.NcnnDetector
import com.aimassist.app.utils.SettingsManager
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import java.util.concurrent.atomic.AtomicLong

/**
 * 屏幕截图和目标检测服务 - ncnn Vulkan 零拷贝管线
 *
 * 核心流程：
 * MediaProjection → ImageReader(HardwareBuffer) → AHardwareBuffer → ncnn Vulkan 推理
 * 全程 GPU，严禁 Bitmap/CPU 缓存
 */
class DetectionService : Service() {

    private val binder = DetectionBinder()
    private val settings = SettingsManager.getInstance()
    private var detector: NcnnDetector? = null

    // 检测线程 (HandlerThread，非协程，减少调度开销)
    private var detectionThread: HandlerThread? = null
    private var detectionHandler: Handler? = null
    private val mainHandler = Handler(android.os.Looper.getMainLooper())

    private var mediaProjection: MediaProjection? = null
    private var virtualDisplay: VirtualDisplay? = null
    private var imageReader: ImageReader? = null

    private val isRunning = AtomicBoolean(false)
    // 线程安全的统计字段
    private val statsFps = AtomicInteger(0)
    private val statsInferenceTimeMs = AtomicLong(0)
    private val statsTotalDetections = AtomicLong(0)
    private val statsTotalClicks = AtomicLong(0)
    private val statsCurrentObjects = AtomicInteger(0)
    private val statsRuntimeMs = AtomicLong(0)
    private var startTime = 0L

    /** 最近一次启动失败的错误信息，主线程可读取 */
    @Volatile
    var lastError: String? = null
        private set

    private var onDetectionCallback: ((List<DetectionResult>, Long) -> Unit)? = null
    private var onStatsCallback: ((InferenceStats) -> Unit)? = null

    // 屏幕参数
    private var screenWidth = 0
    private var screenHeight = 0
    private var screenDensity = 0

    // uinput Root 点击
    private var uinputFd = -1

    companion object {
        private const val TAG = "DetectionService"
        private const val NOTIFICATION_ID = 1001
        const val ACTION_START = "action_start"
        const val ACTION_STOP = "action_stop"
        const val EXTRA_RESULT_CODE = "result_code"
        const val EXTRA_DATA = "data"

        /** 广播 Action：检测启动失败 */
        const val ACTION_START_FAILED = "com.aimassist.app.ACTION_START_FAILED"
        const val EXTRA_ERROR_MESSAGE = "error_message"

        private var instance: DetectionService? = null

        fun isRunning(): Boolean = instance?.isRunning?.get() ?: false
        fun getInstance(): DetectionService? = instance
    }

    inner class DetectionBinder : Binder() {
        fun getService(): DetectionService = this@DetectionService
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        Log.i(TAG, "DetectionService created")
    }

    override fun onBind(intent: Intent): IBinder = binder

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent == null) {
            // 系统重启了服务但 MediaProjection token 已失效
            // 必须先 startForeground 再 stopSelf，否则 ANR
            startForeground(NOTIFICATION_ID, createNotification())
            stopSelf()
            return START_NOT_STICKY
        }
        when (intent.action) {
            ACTION_START -> {
                // 必须立即 startForeground，否则 startForegroundService 后 5 秒 ANR
                startForeground(NOTIFICATION_ID, createNotification())

                val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, -1)
                val data: Intent? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                    intent.getParcelableExtra(EXTRA_DATA, Intent::class.java)
                } else {
                    @Suppress("DEPRECATION")
                    intent.getParcelableExtra(EXTRA_DATA)
                }

                Log.i(TAG, "ACTION_START: resultCode=$resultCode, data=${data != null}")

                if (resultCode != -1 && data != null) {
                    startDetection(resultCode, data)
                } else {
                    broadcastError("无效的 MediaProjection 参数 (resultCode=$resultCode, data=${data != null})")
                    stopForeground(STOP_FOREGROUND_REMOVE)
                    stopSelf()
                }
            }
            ACTION_STOP -> stopDetection()
        }
        return START_NOT_STICKY
    }

    private fun startDetection(resultCode: Int, data: Intent) {
        if (isRunning.get()) return

        if (!initializeDetector()) {
            val msg = "检测器初始化失败: 请检查模型文件是否正确"
            Log.e(TAG, msg)
            broadcastError(msg)
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return
        }

        if (!initializeMediaProjection(resultCode, data)) {
            val msg = "MediaProjection 初始化失败: 请重新授权屏幕录制"
            Log.e(TAG, msg)
            broadcastError(msg)
            detector?.release()
            detector = null
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return
        }

        // 初始化 Root 点击 (如果启用)
        if (settings.clickMode == 1) {
            initializeUinput()
        }

        isRunning.set(true)
        startTime = System.nanoTime()
        statsRuntimeMs.set(0)
        lastError = null

        // 通知已在 onStartCommand 中调用 startForeground
        startDetectionLoop()

        Log.i(TAG, "检测已启动 (ncnn Vulkan)")
    }

    private fun initializeDetector(): Boolean {
        val paramPath = settings.paramFilePath ?: return false
        val binPath = settings.binFilePath ?: return false

        if (!File(paramPath).exists()) {
            Log.e(TAG, "Param 文件不存在: $paramPath")
            return false
        }
        if (!File(binPath).exists()) {
            Log.e(TAG, "Bin 文件不存在: $binPath")
            return false
        }

        detector = NcnnDetector()
        return detector!!.initialize(paramPath, binPath, settings.useGpu)
    }

    private fun initializeMediaProjection(resultCode: Int, data: Intent): Boolean {
        val projectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        mediaProjection = projectionManager.getMediaProjection(resultCode, data)

        mediaProjection?.let { projection ->
            val windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager

            val metrics = DisplayMetrics()
            @Suppress("DEPRECATION")
            windowManager.defaultDisplay.getRealMetrics(metrics)

            screenWidth = metrics.widthPixels
            screenHeight = metrics.heightPixels
            screenDensity = metrics.densityDpi

            // ImageReader: 使用 GPU 友好的配置
            // 注意：不使用 USAGE_GPU_SAMPLED_IMAGE 因为需要兼容 API 26+
            // HardwareBuffer 在 API 28+ 可用
            imageReader = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                // Android 10+: 使用 HardwareBuffer 标志
                ImageReader.newInstance(
                    screenWidth, screenHeight,
                    PixelFormat.RGBA_8888, 2,
                    HardwareBuffer.USAGE_GPU_SAMPLED_IMAGE or HardwareBuffer.USAGE_CPU_READ_RARELY
                )
            } else {
                ImageReader.newInstance(
                    screenWidth, screenHeight,
                    PixelFormat.RGBA_8888, 2
                )
            }

            virtualDisplay = projection.createVirtualDisplay(
                "AimAssistCapture",
                screenWidth, screenHeight, screenDensity,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader?.surface, null, null
            )

            Log.i(TAG, "MediaProjection 初始化: ${screenWidth}x${screenHeight} @${screenDensity}dpi")
            return true
        }

        return false
    }

    private fun initializeUinput() {
        try {
            detector?.let { det ->
                uinputFd = det.nativeUinputCreate(screenWidth, screenHeight)
                if (uinputFd >= 0) {
                    Log.i(TAG, "uinput 虚拟触摸屏已创建: fd=$uinputFd")
                } else {
                    Log.w(TAG, "uinput 创建失败 (需要 Root 权限)")
                }
            }
        } catch (e: Exception) {
            Log.w(TAG, "uinput 初始化异常: ${e.message}")
        }
    }

    private fun startDetectionLoop() {
        detectionThread = HandlerThread("DetectionThread").apply {
            start()
        }
        detectionHandler = Handler(detectionThread!!.looper)

        detectionHandler?.post(object : Runnable {
            private var frameCount = 0
            private var fpsStartTime = System.nanoTime()

            override fun run() {
                if (!isRunning.get()) return

                // 获取最新帧
                val image: Image? = try {
                    imageReader?.acquireLatestImage()
                } catch (e: Exception) {
                    null
                }

                if (image != null) {
                    processFrame(image)
                    image.close()

                    // FPS 计算
                    frameCount++
                    val elapsed = (System.nanoTime() - fpsStartTime) / 1_000_000L
                    if (elapsed >= 1000) {
                        statsFps.set(frameCount)
                        frameCount = 0
                        fpsStartTime = System.nanoTime()
                    }

                    statsRuntimeMs.set((System.nanoTime() - startTime) / 1_000_000L)
                }

                // 调度下一帧
                if (isRunning.get()) {
                    detectionHandler?.post(this)
                }
            }
        })
    }

    private fun processFrame(image: Image) {
        val det = detector ?: return

        // 获取 HardwareBuffer (零拷贝路径)
        val hardwareBuffer: HardwareBuffer? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            image.hardwareBuffer
        } else {
            null
        }

        if (hardwareBuffer == null) {
            Log.w(TAG, "无法获取 HardwareBuffer (需要 Android 9+)")
            return
        }

        try {
            // ncnn 推理 (通过 JNI 传递 HardwareBuffer)
            val results = det.detect(
                hardwareBuffer,
                image.width, image.height,
                settings.confidenceThreshold,
                settings.nmsThreshold
            )

            // 获取耗时
            val timings = det.getTimings()
            val inferenceTime = if (timings.size >= 5) timings[4].toLong() else 0L

            statsInferenceTimeMs.set(inferenceTime)
            statsCurrentObjects.set(results.size)
            statsTotalDetections.addAndGet(results.size.toLong())

            // 回调到主线程
            val snapshot = buildStatsSnapshot()
            mainHandler.post {
                onDetectionCallback?.invoke(results, inferenceTime)
                onStatsCallback?.invoke(snapshot)
            }

            // 自动点击
            if (settings.enableAutoClick && results.isNotEmpty()) {
                processAutoClick(results)
            }
        } finally {
            hardwareBuffer.close()
        }
    }

    private fun processAutoClick(results: List<DetectionResult>) {
        val targetResults = if (settings.targetClass.isNullOrEmpty()) {
            results
        } else {
            results.filter { it.label.equals(settings.targetClass, ignoreCase = true) }
        }

        if (targetResults.isEmpty()) return

        val target = targetResults.maxByOrNull { it.confidence } ?: return

        val clickX = target.getCenterX() + settings.clickOffsetX
        val clickY = target.getCenterY() + settings.clickOffsetY

        when (settings.clickMode) {
            0 -> {
                // 无障碍服务模式
                if (settings.clickDelay > 0) {
                    detectionHandler?.postDelayed({
                        AutoClickService.click(clickX, clickY, 0)
                    }, settings.clickDelay)
                } else {
                    AutoClickService.click(clickX, clickY, 0)
                }
                statsTotalClicks.incrementAndGet()
            }
            1 -> {
                // Root uinput 模式
                if (uinputFd >= 0) {
                    detector?.nativeUinputTap(clickX.toInt(), clickY.toInt())
                    statsTotalClicks.incrementAndGet()
                }
            }
        }
    }

    private fun stopDetection() {
        isRunning.set(false)

        detectionHandler?.removeCallbacksAndMessages(null)
        detectionThread?.quitSafely()
        detectionThread = null
        detectionHandler = null

        virtualDisplay?.release()
        virtualDisplay = null

        imageReader?.close()
        imageReader = null

        mediaProjection?.stop()
        mediaProjection = null

        // 释放 uinput
        if (uinputFd >= 0) {
            detector?.nativeUinputDestroy()
            uinputFd = -1
        }

        detector?.release()
        detector = null

        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()

        Log.i(TAG, "检测已停止")
    }

    private fun createNotification(): Notification {
        val pendingIntent = PendingIntent.getActivity(
            this, 0,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE
        )

        return NotificationCompat.Builder(this, App.CHANNEL_DETECTION)
            .setContentTitle("AimAssist 检测服务运行中")
            .setContentText("ncnn Vulkan 推理中...")
            .setSmallIcon(R.drawable.ic_target)
            .setContentIntent(pendingIntent)
            .setOngoing(true)
            .build()
    }

    fun setDetectionCallback(callback: (List<DetectionResult>, Long) -> Unit) {
        onDetectionCallback = callback
    }

    fun setStatsCallback(callback: (InferenceStats) -> Unit) {
        onStatsCallback = callback
    }

    fun getStats(): InferenceStats = buildStatsSnapshot()

    private fun buildStatsSnapshot(): InferenceStats {
        return InferenceStats(
            fps = statsFps.get(),
            inferenceTimeMs = statsInferenceTimeMs.get(),
            totalDetections = statsTotalDetections.get(),
            totalClicks = statsTotalClicks.get(),
            currentObjects = statsCurrentObjects.get(),
            runtimeMs = statsRuntimeMs.get()
        )
    }

    private fun broadcastError(message: String) {
        lastError = message
        Log.e(TAG, "广播错误: $message")
        val intent = Intent(ACTION_START_FAILED).apply {
            putExtra(EXTRA_ERROR_MESSAGE, message)
            setPackage(packageName)
        }
        sendBroadcast(intent)
    }

    override fun onDestroy() {
        super.onDestroy()
        stopDetection()
        instance = null
    }
}
