// AimAssist JNI 桥接 - 参考 nihui/ncnn-android-yolov8
// 去掉摄像头，使用 AHardwareBuffer (屏幕截图) 输入

#include <jni.h>
#include <android/log.h>
#include <android/hardware_buffer.h>
#include <android/hardware_buffer_jni.h>
#include <string>
#include <vector>

#include <net.h>
#include <gpu.h>
#include <benchmark.h>

#include "yolov8.h"
#include "uinput_injector.h"

#define TAG "AimAssistJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static aimassist::YOLOv8* g_yolov8 = 0;
static aimassist::UinputInjector* g_uinput = 0;
static ncnn::Mutex lock;

// 性能计时
static float g_timings[5] = {0}; // capture, preprocess, inference, postprocess, total

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    LOGI("JNI_OnLoad: 初始化 ncnn GPU");
    ncnn::create_gpu_instance();

    int gpu_count = ncnn::get_gpu_count();
    LOGI("检测到 %d 个 Vulkan GPU", gpu_count);
    if (gpu_count > 0)
    {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(0);
        LOGI("GPU 0: %s", info.device_name());
    }

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    LOGI("JNI_OnUnload: 销毁 ncnn GPU");
    {
        ncnn::MutexLockGuard g(lock);
        delete g_yolov8;
        g_yolov8 = 0;
    }
    if (g_uinput)
    {
        delete g_uinput;
        g_uinput = 0;
    }
    ncnn::destroy_gpu_instance();
}

// ============================================================================
// 模型加载 - 从文件系统路径
// ============================================================================

JNIEXPORT jboolean JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeInit(
    JNIEnv* env, jobject thiz,
    jstring param_path, jstring bin_path, jboolean use_gpu)
{
    const char* param = env->GetStringUTFChars(param_path, nullptr);
    const char* bin = env->GetStringUTFChars(bin_path, nullptr);

    LOGI("nativeInit: param=%s, bin=%s, gpu=%d", param, bin, (int)use_gpu);

    {
        ncnn::MutexLockGuard g(lock);

        if (g_yolov8)
        {
            delete g_yolov8;
            g_yolov8 = 0;
        }

        // 切换 GPU 时需要重建 GPU 实例 (参考 nihui)
        ncnn::destroy_gpu_instance();
        if (use_gpu)
            ncnn::create_gpu_instance();

        g_yolov8 = new aimassist::YOLOv8;
        int ret = g_yolov8->load(param, bin, use_gpu);

        if (ret != 0)
        {
            LOGE("模型加载失败: ret=%d", ret);
            delete g_yolov8;
            g_yolov8 = 0;
        }
    }

    env->ReleaseStringUTFChars(param_path, param);
    env->ReleaseStringUTFChars(bin_path, bin);

    return g_yolov8 != 0 ? JNI_TRUE : JNI_FALSE;
}

// ============================================================================
// 检测 - 输入 HardwareBuffer (屏幕截图)
// ============================================================================

JNIEXPORT jobjectArray JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeDetect(
    JNIEnv* env, jobject thiz,
    jobject hardware_buffer, jint width, jint height,
    jfloat conf_threshold, jfloat nms_threshold)
{
    double t_total_start = ncnn::get_current_time();

    // 获取 AHardwareBuffer
    AHardwareBuffer* ahb = AHardwareBuffer_fromHardwareBuffer(env, hardware_buffer);
    if (!ahb)
    {
        LOGE("AHardwareBuffer_fromHardwareBuffer 失败");
        return nullptr;
    }

    // 锁定获取像素数据
    AHardwareBuffer_Desc desc;
    AHardwareBuffer_describe(ahb, &desc);

    void* data = nullptr;
    int ret = AHardwareBuffer_lock(ahb, AHARDWAREBUFFER_USAGE_CPU_READ_RARELY,
                                   -1, nullptr, &data);
    if (ret != 0 || !data)
    {
        LOGE("AHardwareBuffer_lock 失败: %d", ret);
        return nullptr;
    }

    double t_capture = ncnn::get_current_time();
    g_timings[0] = (float)(t_capture - t_total_start);

    // 检测
    std::vector<aimassist::Object> objects;
    {
        ncnn::MutexLockGuard g(lock);
        if (g_yolov8)
        {
            g_yolov8->detect_from_rgba(
                (const unsigned char*)data,
                (int)desc.width, (int)desc.height,
                (int)(desc.stride * 4),  // stride 是像素单位，转字节
                objects,
                conf_threshold, nms_threshold);

            g_timings[2] = g_yolov8->get_inference_time_ms();
        }
    }

    AHardwareBuffer_unlock(ahb, nullptr);

    double t_post_start = ncnn::get_current_time();

    // 构建 Java 返回对象
    jclass obj_cls = env->FindClass("com/aimassist/app/ncnn/NcnnDetector$DetectedObject");
    if (!obj_cls)
    {
        LOGE("找不到 DetectedObject 类");
        return nullptr;
    }

    jmethodID init_id = env->GetMethodID(obj_cls, "<init>", "(FFFFIF)V");
    if (!init_id)
    {
        LOGE("找不到 DetectedObject 构造函数");
        return nullptr;
    }

    jobjectArray result = env->NewObjectArray((int)objects.size(), obj_cls, nullptr);

    for (size_t i = 0; i < objects.size(); i++)
    {
        const auto& obj = objects[i];
        jobject jobj = env->NewObject(obj_cls, init_id,
                                       obj.x, obj.y, obj.w, obj.h,
                                       obj.label, obj.prob);
        env->SetObjectArrayElement(result, (int)i, jobj);
        env->DeleteLocalRef(jobj);
    }

    double t_total_end = ncnn::get_current_time();
    g_timings[1] = 0; // preprocess 包含在 inference_time 中
    g_timings[3] = (float)(t_total_end - t_post_start);
    g_timings[4] = (float)(t_total_end - t_total_start);

    return result;
}

// ============================================================================
// 工具方法
// ============================================================================

JNIEXPORT void JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeRelease(JNIEnv* env, jobject thiz)
{
    LOGI("nativeRelease");
    ncnn::MutexLockGuard g(lock);
    delete g_yolov8;
    g_yolov8 = 0;
}

JNIEXPORT jint JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeGetModelInputSize(JNIEnv* env, jobject thiz)
{
    ncnn::MutexLockGuard g(lock);
    if (g_yolov8 && g_yolov8->is_loaded())
        return g_yolov8->get_target_size();
    return 0;
}

JNIEXPORT jfloatArray JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeGetTimings(JNIEnv* env, jobject thiz)
{
    jfloatArray result = env->NewFloatArray(5);
    env->SetFloatArrayRegion(result, 0, 5, g_timings);
    return result;
}

JNIEXPORT void JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeSetTargetSize(JNIEnv* env, jobject thiz, jint size)
{
    ncnn::MutexLockGuard g(lock);
    if (g_yolov8)
        g_yolov8->set_target_size((int)size);
}

// ============================================================================
// uinput API
// ============================================================================

JNIEXPORT jint JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeUinputCreate(
    JNIEnv* env, jobject thiz, jint screen_w, jint screen_h)
{
    if (g_uinput) { delete g_uinput; g_uinput = 0; }
    g_uinput = new aimassist::UinputInjector();
    return g_uinput->create(screen_w, screen_h);
}

JNIEXPORT jint JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeUinputTap(
    JNIEnv* env, jobject thiz, jint x, jint y)
{
    if (!g_uinput || !g_uinput->is_created()) return -1;
    return g_uinput->tap(x, y);
}

JNIEXPORT void JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeUinputDestroy(JNIEnv* env, jobject thiz)
{
    if (g_uinput) { delete g_uinput; g_uinput = 0; }
}

} // extern "C"
