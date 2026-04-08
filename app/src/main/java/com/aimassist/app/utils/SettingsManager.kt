package com.aimassist.app.utils

import android.content.Context
import android.content.SharedPreferences
import com.aimassist.app.App

/**
 * 设置管理器 - ncnn 版本
 */
class SettingsManager private constructor() {
    private val prefs: SharedPreferences = App.instance.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    var confidenceThreshold: Float
        get() = prefs.getFloat(KEY_CONFIDENCE, 0.5f)
        set(value) = prefs.edit().putFloat(KEY_CONFIDENCE, value).apply()

    var nmsThreshold: Float
        get() = prefs.getFloat(KEY_NMS, 0.45f)
        set(value) = prefs.edit().putFloat(KEY_NMS, value).apply()

    var inputSize: Int
        get() = prefs.getInt(KEY_INPUT_SIZE, 640)
        set(value) = prefs.edit().putInt(KEY_INPUT_SIZE, value).apply()

    /** ncnn .param 文件路径 */
    var paramFilePath: String?
        get() = prefs.getString(KEY_PARAM_PATH, null)
        set(value) = prefs.edit().putString(KEY_PARAM_PATH, value).apply()

    /** ncnn .bin 文件路径 */
    var binFilePath: String?
        get() = prefs.getString(KEY_BIN_PATH, null)
        set(value) = prefs.edit().putString(KEY_BIN_PATH, value).apply()

    var enableAutoClick: Boolean
        get() = prefs.getBoolean(KEY_AUTO_CLICK, false)
        set(value) = prefs.edit().putBoolean(KEY_AUTO_CLICK, value).apply()

    var clickDelay: Long
        get() = prefs.getLong(KEY_CLICK_DELAY, 0)
        set(value) = prefs.edit().putLong(KEY_CLICK_DELAY, value).apply()

    var clickOffsetX: Int
        get() = prefs.getInt(KEY_OFFSET_X, 0)
        set(value) = prefs.edit().putInt(KEY_OFFSET_X, value).apply()

    var clickOffsetY: Int
        get() = prefs.getInt(KEY_OFFSET_Y, 0)
        set(value) = prefs.edit().putInt(KEY_OFFSET_Y, value).apply()

    var targetClass: String?
        get() = prefs.getString(KEY_TARGET_CLASS, null)
        set(value) = prefs.edit().putString(KEY_TARGET_CLASS, value).apply()

    var useGpu: Boolean
        get() = prefs.getBoolean(KEY_USE_GPU, true)
        set(value) = prefs.edit().putBoolean(KEY_USE_GPU, value).apply()

    /** 点击模式: 0=无障碍服务, 1=Root uinput */
    var clickMode: Int
        get() = prefs.getInt(KEY_CLICK_MODE, 0)
        set(value) = prefs.edit().putInt(KEY_CLICK_MODE, value).apply()

    fun getModelName(): String {
        val path = paramFilePath ?: return "未加载模型"
        return path.substringAfterLast("/")
            .removeSuffix(".param")
            .removeSuffix(".ncnn")
    }

    fun isModelLoaded(): Boolean {
        return !paramFilePath.isNullOrEmpty() && !binFilePath.isNullOrEmpty()
    }

    companion object {
        private const val PREFS_NAME = "aim_assist_settings"
        private const val KEY_CONFIDENCE = "confidence_threshold"
        private const val KEY_NMS = "nms_threshold"
        private const val KEY_INPUT_SIZE = "input_size"
        private const val KEY_PARAM_PATH = "param_file_path"
        private const val KEY_BIN_PATH = "bin_file_path"
        private const val KEY_AUTO_CLICK = "auto_click"
        private const val KEY_CLICK_DELAY = "click_delay"
        private const val KEY_OFFSET_X = "offset_x"
        private const val KEY_OFFSET_Y = "offset_y"
        private const val KEY_TARGET_CLASS = "target_class"
        private const val KEY_USE_GPU = "use_gpu"
        private const val KEY_CLICK_MODE = "click_mode"

        @Volatile
        private var instance: SettingsManager? = null

        fun getInstance(): SettingsManager {
            return instance ?: synchronized(this) {
                instance ?: SettingsManager().also { instance = it }
            }
        }
    }
}
