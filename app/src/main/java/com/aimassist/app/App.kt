package com.aimassist.app

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.os.Build

class App : Application() {

    override fun onCreate() {
        super.onCreate()
        instance = this
        createNotificationChannels()
    }

    private fun createNotificationChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channels = listOf(
                NotificationChannel(
                    CHANNEL_DETECTION,
                    "Detection Service",
                    NotificationManager.IMPORTANCE_LOW
                ).apply {
                    description = "目标检测服务运行通知"
                    setShowBadge(false)
                },
                NotificationChannel(
                    CHANNEL_AUTO_CLICK,
                    "Auto Click",
                    NotificationManager.IMPORTANCE_MIN
                ).apply {
                    description = "自动点击服务"
                    setShowBadge(false)
                }
            )

            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.createNotificationChannels(channels)
        }
    }

    companion object {
        lateinit var instance: App
            private set

        const val CHANNEL_DETECTION = "detection_service"
        const val CHANNEL_AUTO_CLICK = "auto_click_service"
    }
}
