package com.aimassist.app

import android.Manifest
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.os.IBinder
import android.provider.Settings
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.aimassist.app.data.InferenceStats
import com.aimassist.app.databinding.ActivityMainBinding
import com.aimassist.app.service.AutoClickService
import com.aimassist.app.service.DetectionService
import com.aimassist.app.utils.SettingsManager
import kotlinx.coroutines.*

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val settings = SettingsManager.getInstance()
    private var detectionService: DetectionService? = null
    private var isBound = false

    private var uiUpdateJob: Job? = null

    private val mediaProjectionLauncher = registerForActivityResult(
        ActivityResultContracts.StartActivityForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK && result.data != null) {
            startDetectionService(result.resultCode, result.data!!)
        } else {
            Toast.makeText(this, "需要屏幕录制权限", Toast.LENGTH_SHORT).show()
        }
    }

    private val storagePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val allGranted = permissions.entries.all { it.value }
        if (!allGranted) {
            Toast.makeText(this, "需要存储权限来选择模型文件", Toast.LENGTH_SHORT).show()
        }
    }

    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, service: IBinder?) {
            val binder = service as DetectionService.DetectionBinder
            detectionService = binder.getService()
            isBound = true

            detectionService?.setStatsCallback { stats ->
                updateUI(stats)
            }
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            detectionService = null
            isBound = false
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initViews()
        checkPermissions()
        updateModelInfo()
    }

    override fun onResume() {
        super.onResume()
        updateUIState()
        updateModelInfo()
    }

    private fun initViews() {
        binding.btnSettings.setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
        }

        binding.btnStart.setOnClickListener {
            if (checkRequirements()) {
                requestMediaProjection()
            }
        }

        binding.btnStop.setOnClickListener {
            stopDetection()
        }

        binding.sliderConfidence.addOnChangeListener { _, value, _ ->
            settings.confidenceThreshold = value
            binding.tvConfidenceValue.text = String.format("%.2f", value)
        }
        binding.sliderConfidence.value = settings.confidenceThreshold
        binding.tvConfidenceValue.text = String.format("%.2f", settings.confidenceThreshold)

        binding.switchAutoClick.isChecked = settings.enableAutoClick
        binding.switchAutoClick.setOnCheckedChangeListener { _, isChecked ->
            settings.enableAutoClick = isChecked
            if (isChecked && settings.clickMode == 0 && !AutoClickService.isRunning()) {
                showAccessibilityDialog()
            }
        }
    }

    private fun checkPermissions() {
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            val permissions = arrayOf(
                Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.READ_EXTERNAL_STORAGE
            )
            if (permissions.any {
                    ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
                }) {
                storagePermissionLauncher.launch(permissions)
            }
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
                != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.POST_NOTIFICATIONS),
                    100
                )
            }
        }
    }

    private fun checkRequirements(): Boolean {
        if (!settings.isModelLoaded()) {
            Toast.makeText(this, "请先选择模型文件 (.param + .bin)", Toast.LENGTH_SHORT).show()
            startActivity(Intent(this, SettingsActivity::class.java))
            return false
        }

        if (settings.enableAutoClick && settings.clickMode == 0 && !AutoClickService.isRunning()) {
            showAccessibilityDialog()
            return false
        }

        return true
    }

    private fun requestMediaProjection() {
        val projectionManager = getSystemService(Context.MEDIA_PROJECTION_SERVICE) as MediaProjectionManager
        mediaProjectionLauncher.launch(projectionManager.createScreenCaptureIntent())
    }

    private fun startDetectionService(resultCode: Int, data: Intent) {
        val intent = Intent(this, DetectionService::class.java).apply {
            action = DetectionService.ACTION_START
            putExtra(DetectionService.EXTRA_RESULT_CODE, resultCode)
            putExtra(DetectionService.EXTRA_DATA, data)
        }

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(intent)
        } else {
            startService(intent)
        }

        bindService(intent, serviceConnection, Context.BIND_AUTO_CREATE)

        updateUIState(running = true)
        Toast.makeText(this, R.string.service_started, Toast.LENGTH_SHORT).show()
    }

    private fun stopDetection() {
        val intent = Intent(this, DetectionService::class.java).apply {
            action = DetectionService.ACTION_STOP
        }
        startService(intent)

        if (isBound) {
            unbindService(serviceConnection)
            isBound = false
        }

        detectionService = null
        updateUIState(running = false)
        Toast.makeText(this, R.string.service_stopped, Toast.LENGTH_SHORT).show()
    }

    private fun updateUIState(running: Boolean = DetectionService.isRunning()) {
        if (running) {
            binding.btnStart.visibility = View.GONE
            binding.btnStop.visibility = View.VISIBLE
            binding.tvStatus.text = getString(R.string.status_running)
            binding.statusIndicator.setBackgroundResource(R.drawable.circle_status_active)

            startUIUpdates()
        } else {
            binding.btnStart.visibility = View.VISIBLE
            binding.btnStop.visibility = View.GONE
            binding.tvStatus.text = getString(R.string.status_idle)
            binding.statusIndicator.setBackgroundResource(R.drawable.circle_status_inactive)

            stopUIUpdates()
            resetStats()
        }
    }

    private fun startUIUpdates() {
        uiUpdateJob?.cancel()
        uiUpdateJob = lifecycleScope.launch {
            while (isActive) {
                detectionService?.getStats()?.let { stats ->
                    updateUI(stats)
                }
                delay(100)
            }
        }
    }

    private fun stopUIUpdates() {
        uiUpdateJob?.cancel()
        uiUpdateJob = null
    }

    private fun updateUI(stats: InferenceStats) {
        binding.tvFps.text = stats.fps.toString()
        binding.tvObjects.text = stats.currentObjects.toString()
        binding.tvInferenceTime.text = "${stats.inferenceTimeMs}ms"
        binding.tvTotalClicks.text = stats.totalClicks.toString()
    }

    private fun resetStats() {
        binding.tvFps.text = "0"
        binding.tvObjects.text = "0"
        binding.tvInferenceTime.text = "0ms"
    }

    private fun updateModelInfo() {
        val modelName = settings.getModelName()
        binding.tvModelName.text = modelName

        if (settings.isModelLoaded()) {
            val engine = "ncnn Vulkan"
            val gpu = if (settings.useGpu) "GPU" else "CPU"
            binding.tvModelInfo.text = "引擎: $engine | 输入: ${settings.inputSize} | $gpu"
        } else {
            binding.tvModelInfo.text = getString(R.string.select_model)
        }
    }

    private fun showAccessibilityDialog() {
        AlertDialog.Builder(this)
            .setTitle("需要无障碍权限")
            .setMessage("自动点击功能需要开启无障碍服务，请前往设置开启")
            .setPositiveButton("前往设置") { _, _ ->
                val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS).apply {
                    flags = Intent.FLAG_ACTIVITY_NEW_TASK
                }
                startActivity(intent)
            }
            .setNegativeButton("取消", null)
            .show()
    }

    override fun onDestroy() {
        super.onDestroy()
        stopUIUpdates()
        if (isBound) {
            unbindService(serviceConnection)
        }
    }
}
