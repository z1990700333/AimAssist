package com.aimassist.app.data

import android.graphics.RectF

/**
 * 检测结果数据类
 */
data class DetectionResult(
    val label: String,
    val confidence: Float,
    val bbox: RectF,
    val classId: Int
) {
    fun getCenterX(): Float = (bbox.left + bbox.right) / 2
    fun getCenterY(): Float = (bbox.top + bbox.bottom) / 2
    fun getWidth(): Float = bbox.right - bbox.left
    fun getHeight(): Float = bbox.bottom - bbox.top
}

/**
 * 推理统计信息
 */
data class InferenceStats(
    var fps: Int = 0,
    var inferenceTimeMs: Long = 0,
    var totalDetections: Long = 0,
    var totalClicks: Long = 0,
    var currentObjects: Int = 0,
    var runtimeMs: Long = 0
)

/**
 * 点击事件
 */
data class ClickEvent(
    val x: Float,
    val y: Float,
    val delay: Long
)
