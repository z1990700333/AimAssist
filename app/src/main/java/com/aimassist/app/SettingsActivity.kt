package com.aimassist.app

import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import com.aimassist.app.databinding.ActivitySettingsBinding
import com.aimassist.app.utils.SettingsManager
import java.io.File
import java.io.FileOutputStream

/**
 * 设置页面 - ncnn 模型配置
 */
class SettingsActivity : AppCompatActivity() {

    private lateinit var binding: ActivitySettingsBinding
    private val settings = SettingsManager.getInstance()

    // 当前选择的文件类型
    private var selectingFileType = FileType.PARAM

    private enum class FileType { PARAM, BIN }

    private val filePickerLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { handleFileSelection(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivitySettingsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        initViews()
        loadSettings()
    }

    private fun initViews() {
        binding.toolbar.setNavigationOnClickListener {
            finish()
        }

        // Param 文件选择
        binding.tilParamFile.hint = "模型参数文件 (.param)"
        binding.etParamFile.setOnClickListener {
            selectingFileType = FileType.PARAM
            filePickerLauncher.launch("application/octet-stream")
        }
        binding.tilParamFile.setEndIconOnClickListener {
            selectingFileType = FileType.PARAM
            filePickerLauncher.launch("application/octet-stream")
        }

        // Bin 文件选择 (恢复显示)
        binding.tilBinFile.visibility = android.view.View.VISIBLE
        binding.tilBinFile.hint = "模型权重文件 (.bin)"
        binding.etBinFile.setOnClickListener {
            selectingFileType = FileType.BIN
            filePickerLauncher.launch("application/octet-stream")
        }
        binding.tilBinFile.setEndIconOnClickListener {
            selectingFileType = FileType.BIN
            filePickerLauncher.launch("application/octet-stream")
        }

        // 保存按钮
        binding.btnSave.setOnClickListener {
            saveSettings()
        }
    }

    private fun loadSettings() {
        binding.etParamFile.setText(settings.paramFilePath ?: "")
        binding.etBinFile.setText(settings.binFilePath ?: "")
        binding.sliderConfidenceSetting.value = settings.confidenceThreshold
        binding.sliderNms.value = settings.nmsThreshold
        binding.etInputSize.setText(settings.inputSize.toString())
        binding.etTargetClass.setText(settings.targetClass ?: "")
        binding.switchGpu.isChecked = settings.useGpu
        binding.etClickDelay.setText(settings.clickDelay.toString())
        binding.etClickOffsetX.setText(settings.clickOffsetX.toString())
        binding.etClickOffsetY.setText(settings.clickOffsetY.toString())
    }

    private fun handleFileSelection(uri: Uri) {
        val fileName = getFileName(uri)

        when (selectingFileType) {
            FileType.PARAM -> {
                if (!fileName.endsWith(".param", ignoreCase = true)) {
                    Toast.makeText(this, "请选择 .param 格式的模型参数文件", Toast.LENGTH_SHORT).show()
                    return
                }
                copyAndSave(uri, fileName) { path ->
                    settings.paramFilePath = path
                    binding.etParamFile.setText(path)
                    Toast.makeText(this, "参数文件已导入", Toast.LENGTH_SHORT).show()
                }
            }
            FileType.BIN -> {
                if (!fileName.endsWith(".bin", ignoreCase = true)) {
                    Toast.makeText(this, "请选择 .bin 格式的模型权重文件", Toast.LENGTH_SHORT).show()
                    return
                }
                copyAndSave(uri, fileName) { path ->
                    settings.binFilePath = path
                    binding.etBinFile.setText(path)
                    Toast.makeText(this, "权重文件已导入", Toast.LENGTH_SHORT).show()
                }
            }
        }
    }

    private fun copyAndSave(uri: Uri, fileName: String, onSuccess: (String) -> Unit) {
        val destFile = File(filesDir, "models/$fileName")
        destFile.parentFile?.mkdirs()

        try {
            contentResolver.openInputStream(uri)?.use { input ->
                FileOutputStream(destFile).use { output ->
                    input.copyTo(output)
                }
            }
            onSuccess(destFile.absolutePath)
        } catch (e: Exception) {
            Toast.makeText(this, "文件保存失败: ${e.message}", Toast.LENGTH_SHORT).show()
        }
    }

    private fun getFileName(uri: Uri): String {
        var result = "model.bin"
        contentResolver.query(uri, null, null, null, null)?.use { cursor ->
            if (cursor.moveToFirst()) {
                val index = cursor.getColumnIndex(android.provider.OpenableColumns.DISPLAY_NAME)
                if (index >= 0) {
                    result = cursor.getString(index)
                }
            }
        }
        return result
    }

    private fun saveSettings() {
        settings.confidenceThreshold = binding.sliderConfidenceSetting.value
        settings.nmsThreshold = binding.sliderNms.value

        val inputSize = binding.etInputSize.text.toString().toIntOrNull() ?: 640
        settings.inputSize = inputSize

        val targetClass = binding.etTargetClass.text.toString().trim()
        settings.targetClass = if (targetClass.isEmpty()) null else targetClass

        settings.useGpu = binding.switchGpu.isChecked

        val clickDelay = binding.etClickDelay.text.toString().toLongOrNull() ?: 0
        settings.clickDelay = clickDelay

        val offsetX = binding.etClickOffsetX.text.toString().toIntOrNull() ?: 0
        settings.clickOffsetX = offsetX

        val offsetY = binding.etClickOffsetY.text.toString().toIntOrNull() ?: 0
        settings.clickOffsetY = offsetY

        Toast.makeText(this, "设置已保存", Toast.LENGTH_SHORT).show()
        finish()
    }
}
