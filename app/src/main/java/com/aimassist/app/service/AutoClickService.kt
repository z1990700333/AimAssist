package com.aimassist.app.service

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.content.Intent
import android.graphics.Path
import android.util.Log
import android.view.accessibility.AccessibilityEvent

/**
 * 自动点击无障碍服务 (非 Root 模式)
 * 点击 duration 极短 (1ms) 以减少延迟
 */
class AutoClickService : AccessibilityService() {

    companion object {
        private const val TAG = "AutoClickService"
        private var instance: AutoClickService? = null

        fun isRunning(): Boolean = instance != null

        /**
         * 执行点击 (极短 duration)
         */
        fun click(x: Float, y: Float, delay: Long = 0): Boolean {
            val service = instance ?: return false
            return service.performClick(x, y)
        }

        /**
         * 执行滑动
         */
        fun swipe(startX: Float, startY: Float, endX: Float, endY: Float, duration: Long = 300): Boolean {
            val service = instance ?: return false
            return service.performSwipe(startX, startY, endX, endY, duration)
        }
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        Log.i(TAG, "AutoClickService connected")
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) {
        // 不需要处理事件
    }

    override fun onInterrupt() {
        Log.i(TAG, "AutoClickService interrupted")
    }

    override fun onUnbind(intent: Intent?): Boolean {
        instance = null
        Log.i(TAG, "AutoClickService unbound")
        return super.onUnbind(intent)
    }

    /**
     * 执行点击手势 - 极短 duration (1ms)
     */
    private fun performClick(x: Float, y: Float): Boolean {
        return try {
            val path = Path().apply {
                moveTo(x, y)
            }

            val gesture = GestureDescription.Builder()
                .addStroke(GestureDescription.StrokeDescription(path, 0, 1)) // 1ms duration
                .build()

            dispatchGesture(gesture, object : GestureResultCallback() {
                override fun onCompleted(gestureDescription: GestureDescription?) {
                    // 点击完成
                }
                override fun onCancelled(gestureDescription: GestureDescription?) {
                    Log.d(TAG, "Click cancelled at ($x, $y)")
                }
            }, null)

            true
        } catch (e: Exception) {
            Log.e(TAG, "Click failed: ${e.message}")
            false
        }
    }

    /**
     * 执行滑动手势
     */
    private fun performSwipe(startX: Float, startY: Float, endX: Float, endY: Float, duration: Long): Boolean {
        return try {
            val path = Path().apply {
                moveTo(startX, startY)
                lineTo(endX, endY)
            }

            val gesture = GestureDescription.Builder()
                .addStroke(GestureDescription.StrokeDescription(path, 0, duration))
                .build()

            dispatchGesture(gesture, object : GestureResultCallback() {
                override fun onCompleted(gestureDescription: GestureDescription?) {}
                override fun onCancelled(gestureDescription: GestureDescription?) {
                    Log.d(TAG, "Swipe cancelled")
                }
            }, null)

            true
        } catch (e: Exception) {
            Log.e(TAG, "Swipe failed: ${e.message}")
            false
        }
    }
}
