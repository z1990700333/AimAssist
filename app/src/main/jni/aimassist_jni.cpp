#include <jni.h>
#include <android/log.h>
#include <android/hardware_buffer_jni.h>
#include <android/asset_manager_jni.h>
#include <string>

#include <net.h>
#include <gpu.h>

#include "ncnn_detector.h"
#include "zero_copy_pipeline.h"
#include "uinput_injector.h"

#define TAG "AimAssistJNI"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

// 全局实例
static aimassist::NcnnDetector* g_detector = nullptr;
static aimassist::ZeroCopyPipeline* g_pipeline = nullptr;
static aimassist::UinputInjector* g_uinput = nullptr;

// 性能计时
static float g_timings[5] = {0}; // capture, preprocess, inference, postprocess, total

// ============================================================================
// JNI 生命周期
// ============================================================================

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    LOGI("JNI_OnLoad: 初始化 ncnn GPU 实例");
    ncnn::create_gpu_instance();

    int gpu_count = ncnn::get_gpu_count();
    LOGI("检测到 %d 个 Vulkan GPU", gpu_count);

    if (gpu_count > 0) {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(0);
        LOGI("GPU 0: %s", info.device_name());
        LOGI("  API 版本: %u.%u.%u",
             VK_VERSION_MAJOR(info.api_version()),
             VK_VERSION_MINOR(info.api_version()),
             VK_VERSION_PATCH(info.api_version()));
        LOGI("  FP16 运算: %s", info.support_fp16_arithmetic() ? "支持" : "不支持");
        LOGI("  INT8 运算: %s", info.support_int8_arithmetic() ? "支持" : "不支持");
    }

    return JNI_VERSION_1_6;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
    LOGI("JNI_OnUnload: 销毁 ncnn GPU 实例");

    if (g_detector) {
        delete g_detector;
        g_detector = nullptr;
    }
    if (g_pipeline) {
        delete g_pipeline;
        g_pipeline = nullptr;
    }
    if (g_uinput) {
        delete g_uinput;
        g_uinput = nullptr;
    }

    ncnn::destroy_gpu_instance();
}

// ============================================================================
// 检测器 API
// ============================================================================

JNIEXPORT jboolean JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeInit(
        JNIEnv* env, jobject thiz,
        jstring param_path, jstring bin_path, jboolean use_gpu) {

    const char* param = env->GetStringUTFChars(param_path, nullptr);
    const char* bin = env->GetStringUTFChars(bin_path, nullptr);

    LOGI("初始化检测器: param=%s, bin=%s, gpu=%d", param, bin, use_gpu);

    // 创建检测器
    if (g_detector) {
        delete g_detector;
    }
    g_detector = new aimassist::NcnnDetector();

    int ret = g_detector->load(param, bin, use_gpu);

    if (ret == 0) {
        // 初始化零拷贝管线
        if (g_pipeline) {
            delete g_pipeline;
        }
        g_pipeline = new aimassist::ZeroCopyPipeline();
        int input_size = g_detector->get_input_size();
        g_pipeline->init(input_size, input_size);

        LOGI("检测器初始化成功: input_size=%d", input_size);
    } else {
        LOGE("检测器初始化失败: ret=%d", ret);
        delete g_detector;
        g_detector = nullptr;
    }

    env->ReleaseStringUTFChars(param_path, param);
    env->ReleaseStringUTFChars(bin_path, bin);

    return ret == 0 ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jobjectArray JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeDetect(
        JNIEnv* env, jobject thiz,
        jobject hardware_buffer, jint width, jint height,
        jfloat conf_threshold, jfloat nms_threshold) {

    if (!g_detector || !g_detector->is_loaded() || !g_pipeline) {
        LOGE("检测器未初始化");
        return nullptr;
    }

    auto t_total_start = std::chrono::high_resolution_clock::now();

    // 获取 AHardwareBuffer
    AHardwareBuffer* ahb = AHardwareBuffer_fromHardwareBuffer(env, hardware_buffer);
    if (!ahb) {
        LOGE("获取 AHardwareBuffer 失败");
        return nullptr;
    }

    // 零拷贝管线：AHB → 预处理后的 ncnn::Mat
    ncnn::Mat input_mat;
    int ret = g_pipeline->process(ahb, width, height, input_mat);
    if (ret != 0) {
        LOGE("零拷贝管线处理失败: %d", ret);
        return nullptr;
    }

    g_timings[0] = g_pipeline->get_capture_time_ms();
    g_timings[1] = g_pipeline->get_preprocess_time_ms();

    // ncnn 推理
    std::vector<aimassist::Object> objects;
    ret = g_detector->detect(input_mat, objects, conf_threshold, nms_threshold);
    if (ret != 0) {
        LOGE("推理失败: %d", ret);
        return nullptr;
    }

    g_timings[2] = g_detector->get_inference_time();

    auto t_post_start = std::chrono::high_resolution_clock::now();

    // 坐标缩放：从模型输入尺寸还原到原始屏幕尺寸
    int input_size = g_detector->get_input_size();
    float scale_x = (float)width / input_size;
    float scale_y = (float)height / input_size;

    // 构建 Java 返回对象数组
    jclass obj_cls = env->FindClass("com/aimassist/app/ncnn/NcnnDetector$DetectedObject");
    if (!obj_cls) {
        LOGE("找不到 DetectedObject 类");
        return nullptr;
    }

    jmethodID init_id = env->GetMethodID(obj_cls, "<init>", "(FFFFIF)V");
    if (!init_id) {
        LOGE("找不到 DetectedObject 构造函数");
        return nullptr;
    }

    jobjectArray result = env->NewObjectArray(objects.size(), obj_cls, nullptr);

    for (size_t i = 0; i < objects.size(); i++) {
        const auto& obj = objects[i];

        // 缩放坐标到原始屏幕尺寸
        float x = obj.x * scale_x;
        float y = obj.y * scale_y;
        float w = obj.w * scale_x;
        float h = obj.h * scale_y;

        jobject jobj = env->NewObject(obj_cls, init_id,
                                       x, y, w, h, obj.label, obj.prob);
        env->SetObjectArrayElement(result, i, jobj);
        env->DeleteLocalRef(jobj);
    }

    auto t_total_end = std::chrono::high_resolution_clock::now();
    g_timings[3] = std::chrono::duration<float, std::milli>(
        t_total_end - t_post_start).count();
    g_timings[4] = std::chrono::duration<float, std::milli>(
        t_total_end - t_total_start).count();

    return result;
}

JNIEXPORT void JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeRelease(
        JNIEnv* env, jobject thiz) {

    LOGI("释放检测器");

    if (g_pipeline) {
        delete g_pipeline;
        g_pipeline = nullptr;
    }
    if (g_detector) {
        delete g_detector;
        g_detector = nullptr;
    }
}

JNIEXPORT jint JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeGetModelInputSize(
        JNIEnv* env, jobject thiz) {

    if (g_detector && g_detector->is_loaded()) {
        return g_detector->get_input_size();
    }
    return 0;
}

JNIEXPORT jfloatArray JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeGetTimings(
        JNIEnv* env, jobject thiz) {

    jfloatArray result = env->NewFloatArray(5);
    env->SetFloatArrayRegion(result, 0, 5, g_timings);
    return result;
}

// ============================================================================
// uinput API
// ============================================================================

JNIEXPORT jint JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeUinputCreate(
        JNIEnv* env, jobject thiz,
        jint screen_w, jint screen_h) {

    LOGI("创建 uinput 虚拟触摸屏: %dx%d", screen_w, screen_h);

    if (g_uinput) {
        delete g_uinput;
    }
    g_uinput = new aimassist::UinputInjector();

    return g_uinput->create(screen_w, screen_h);
}

JNIEXPORT jint JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeUinputTap(
        JNIEnv* env, jobject thiz,
        jint x, jint y) {

    if (!g_uinput || !g_uinput->is_created()) {
        return -1;
    }
    return g_uinput->tap(x, y);
}

JNIEXPORT void JNICALL
Java_com_aimassist_app_ncnn_NcnnDetector_nativeUinputDestroy(
        JNIEnv* env, jobject thiz) {

    if (g_uinput) {
        delete g_uinput;
        g_uinput = nullptr;
    }
}

} // extern "C"
