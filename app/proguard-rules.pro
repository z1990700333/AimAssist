# ProGuard rules for AimAssist

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep NcnnDetector and its inner classes
-keep class com.aimassist.app.ncnn.NcnnDetector {
    *;
}
-keep class com.aimassist.app.ncnn.NcnnDetector$DetectedObject {
    *;
}

# Keep data classes
-keep class com.aimassist.app.data.** {
    *;
}

# Keep services
-keep class com.aimassist.app.service.** {
    *;
}

# Kotlin
-keep class kotlin.** { *; }
-keep class kotlin.Metadata { *; }
-dontwarn kotlin.**
-keepclassmembers class **$WhenMappings {
    <fields>;
}

# Coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
