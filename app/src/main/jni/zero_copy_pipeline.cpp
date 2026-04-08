#include "zero_copy_pipeline.h"
#include <android/hardware_buffer.h>
#include <android/log.h>
#include <cstring>
#include <chrono>

#define TAG "ZeroCopyPipeline"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)

namespace aimassist {

ZeroCopyPipeline::ZeroCopyPipeline() = default;

ZeroCopyPipeline::~ZeroCopyPipeline() {
    release();
}

int ZeroCopyPipeline::init(int target_w, int target_h) {
    target_w_ = target_w;
    target_h_ = target_h;

#if NCNN_VULKAN
    if (ncnn::get_gpu_count() > 0) {
        vkdev_ = ncnn::get_gpu_device(0);
        if (vkdev_) {
            blob_vkallocator_ = new ncnn::VkBlobAllocator(vkdev_);
            staging_vkallocator_ = new ncnn::VkStagingAllocator(vkdev_);
            LOGI("Vulkan 设备就绪，零拷贝管线可用");
        }
    } else {
        LOGW("无 Vulkan GPU，使用 CPU 回退路径");
    }
#endif

    initialized_ = true;
    LOGI("零拷贝管线初始化完成: target=%dx%d", target_w_, target_h_);
    return 0;
}

int ZeroCopyPipeline::process_cpu_fallback(AHardwareBuffer* ahb, int src_w, int src_h,
                                            ncnn::Mat& out) {
    auto t0 = std::chrono::high_resolution_clock::now();

    // CPU 回退：锁定 AHB 获取像素数据
    AHardwareBuffer_Desc desc;
    AHardwareBuffer_describe(ahb, &desc);

    void* data = nullptr;
    int ret = AHardwareBuffer_lock(ahb, AHARDWAREBUFFER_USAGE_CPU_READ_RARELY,
                                    -1, nullptr, &data);
    if (ret != 0 || !data) {
        LOGE("AHardwareBuffer_lock 失败: %d", ret);
        return -1;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    capture_time_ms_ = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // RGBA → RGB + resize + normalize
    preprocess_cpu((const uint8_t*)data, desc.width, desc.height, desc.stride, out);

    AHardwareBuffer_unlock(ahb, nullptr);

    auto t2 = std::chrono::high_resolution_clock::now();
    preprocess_time_ms_ = std::chrono::duration<float, std::milli>(t2 - t1).count();

    return 0;
}

int ZeroCopyPipeline::process(AHardwareBuffer* ahb, int src_w, int src_h, ncnn::Mat& out) {
    // 当前使用 CPU 回退路径
    // TODO: 实现完整的 AHB → VkImage → ncnn VkMat 零拷贝路径
    // 需要通过 ncnn VulkanDevice 获取 VkDevice，然后使用
    // VK_ANDROID_external_memory_android_hardware_buffer 扩展导入 AHB
    return process_cpu_fallback(ahb, src_w, src_h, out);
}

int ZeroCopyPipeline::preprocess_cpu(const uint8_t* rgba_data, int src_w, int src_h,
                                      int stride, ncnn::Mat& out) {
    // RGBA → RGB + resize to target_w_ x target_h_ + normalize [0,1]
    // 使用 ncnn::Mat::from_pixels_resize 高效实现

    // stride 是字节为单位的行跨度
    if ((int)(stride / 4) == src_w) {
        // 无 padding，直接转换
        out = ncnn::Mat::from_pixels_resize(
            rgba_data, ncnn::Mat::PIXEL_RGBA2RGB,
            src_w, src_h, target_w_, target_h_);
    } else {
        // 有 padding，需要逐行处理
        std::vector<uint8_t> clean_data(src_w * src_h * 4);
        for (int y = 0; y < src_h; y++) {
            memcpy(clean_data.data() + y * src_w * 4,
                   rgba_data + y * stride,
                   src_w * 4);
        }
        out = ncnn::Mat::from_pixels_resize(
            clean_data.data(), ncnn::Mat::PIXEL_RGBA2RGB,
            src_w, src_h, target_w_, target_h_);
    }

    // 归一化: [0, 255] → [0, 1]
    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    out.substract_mean_normalize(nullptr, norm_vals);

    return 0;
}

void ZeroCopyPipeline::release() {
#if NCNN_VULKAN
    if (blob_vkallocator_) {
        delete blob_vkallocator_;
        blob_vkallocator_ = nullptr;
    }
    if (staging_vkallocator_) {
        delete staging_vkallocator_;
        staging_vkallocator_ = nullptr;
    }
#endif

    vkdev_ = nullptr;
    initialized_ = false;
    LOGI("零拷贝管线已释放");
}

} // namespace aimassist
