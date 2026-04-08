package com.aimassist.app.ncnn

import android.graphics.RectF
import android.hardware.HardwareBuffer
import android.util.Log
import com.aimassist.app.data.DetectionResult

/**
 * ncnn 检测器 JNI 封装
 * 基于 ncnn Vulkan 后端的高性能目标检测
 */
class NcnnDetector {

    /**
     * JNI 返回的检测对象
     */
    data class DetectedObject(
        val x: Float,      // bbox left
        val y: Float,      // bbox top
        val w: Float,      // bbox width
        val h: Float,      // bbox height
        val label: Int,    // 类别 ID
        val prob: Float    // 置信度
    )

    companion object {
        private const val TAG = "NcnnDetector"

        init {
            System.loadLibrary("aimassist")
            Log.i(TAG, "libaimassist.so 已加载")
        }

        val COCO_LABELS = listOf(
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        )
    }

    @Volatile
    private var initialized = false

    // ========================================================================
    // Native 方法声明
    // ========================================================================

    external fun nativeInit(paramPath: String, binPath: String, useGpu: Boolean): Boolean
    external fun nativeDetect(
        hardwareBuffer: HardwareBuffer,
        width: Int, height: Int,
        confThreshold: Float, nmsThreshold: Float
    ): Array<DetectedObject>?
    external fun nativeRelease()
    external fun nativeGetModelInputSize(): Int
    external fun nativeGetTimings(): FloatArray

    // uinput
    external fun nativeUinputCreate(screenW: Int, screenH: Int): Int
    external fun nativeUinputTap(x: Int, y: Int): Int
    external fun nativeUinputDestroy()

    // ========================================================================
    // 高层 API
    // ========================================================================

    /**
     * 初始化检测器
     * @param paramPath .param 文件路径
     * @param binPath .bin 文件路径
     * @param useGpu 是否使用 Vulkan GPU
     */
    fun initialize(paramPath: String, binPath: String, useGpu: Boolean = true): Boolean {
        return try {
            val result = nativeInit(paramPath, binPath, useGpu)
            initialized = result
            if (result) {
                Log.i(TAG, "检测器初始化成功: inputSize=${getModelInputSize()}")
            } else {
                Log.e(TAG, "检测器初始化失败")
            }
            result
        } catch (e: Exception) {
            Log.e(TAG, "初始化异常: ${e.message}")
            false
        }
    }

    /**
     * 执行检测
     * @param hardwareBuffer 屏幕截图的 HardwareBuffer (零拷贝)
     * @param width 图像宽度
     * @param height 图像高度
     * @param confThreshold 置信度阈值
     * @param nmsThreshold NMS 阈值
     * @return 检测结果列表
     */
    fun detect(
        hardwareBuffer: HardwareBuffer,
        width: Int, height: Int,
        confThreshold: Float = 0.5f,
        nmsThreshold: Float = 0.45f
    ): List<DetectionResult> {
        if (!initialized) return emptyList()

        return try {
            val objects = nativeDetect(hardwareBuffer, width, height,
                confThreshold, nmsThreshold) ?: return emptyList()

            objects.map { obj ->
                val label = if (obj.label < COCO_LABELS.size) {
                    COCO_LABELS[obj.label]
                } else {
                    "Class ${obj.label}"
                }
                DetectionResult(
                    label = label,
                    confidence = obj.prob,
                    bbox = RectF(obj.x, obj.y, obj.x + obj.w, obj.y + obj.h),
                    classId = obj.label
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "检测异常: ${e.message}")
            emptyList()
        }
    }

    /**
     * 获取模型输入尺寸
     */
    fun getModelInputSize(): Int {
        return if (initialized) nativeGetModelInputSize() else 0
    }

    /**
     * 获取各阶段耗时 [capture, preprocess, inference, postprocess, total] (ms)
     */
    fun getTimings(): FloatArray {
        return if (initialized) nativeGetTimings() else FloatArray(5)
    }

    /**
     * 释放资源
     */
    fun release() {
        if (initialized) {
            nativeRelease()
            initialized = false
            Log.i(TAG, "检测器已释放")
        }
    }

    fun isInitialized(): Boolean = initialized
}
