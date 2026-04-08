#pragma once

#include <android/hardware_buffer.h>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_android.h>
#include <net.h>
#include <gpu.h>
#include <mat.h>
#include <unordered_map>
#include <chrono>

namespace aimassist {

/**
 * 零拷贝 GPU 管线
 * AHardwareBuffer → VkImage → ncnn::VkMat
 * 全程 GPU，无 CPU 拷贝
 */
class ZeroCopyPipeline {
public:
    ZeroCopyPipeline();
    ~ZeroCopyPipeline();

    /**
     * 初始化管线
     * @param target_w 模型输入宽度
     * @param target_h 模型输入高度
     * @return 成功返回 0
     */
    int init(int target_w, int target_h);

    /**
     * 处理一帧：AHardwareBuffer → ncnn::Mat (CPU 路径回退)
     * 当 Vulkan 外部内存扩展不可用时使用
     */
    int process_cpu_fallback(AHardwareBuffer* ahb, int src_w, int src_h, ncnn::Mat& out);

    /**
     * 处理一帧：AHardwareBuffer → ncnn::VkMat (零拷贝 GPU 路径)
     */
    int process_gpu(AHardwareBuffer* ahb, int src_w, int src_h, ncnn::VkMat& out);

    /**
     * 处理一帧：自动选择最优路径
     * 返回 ncnn::Mat（如果 GPU 路径可用，内部仍走 GPU，最后拷贝回 CPU Mat）
     */
    int process(AHardwareBuffer* ahb, int src_w, int src_h, ncnn::Mat& out);

    /**
     * 获取各阶段耗时
     */
    float get_capture_time_ms() const { return capture_time_ms_; }
    float get_preprocess_time_ms() const { return preprocess_time_ms_; }

    void release();

private:
    // Vulkan 外部内存导入
    struct VkAHBImage {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView image_view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        int width = 0;
        int height = 0;
    };

    int import_ahb_to_vulkan(AHardwareBuffer* ahb, VkAHBImage& out_image);
    void release_vk_ahb_image(VkAHBImage& img);

    // Vulkan 预处理：RGBA → RGB + resize + normalize
    int preprocess_vulkan(const VkAHBImage& src, int src_w, int src_h, ncnn::VkMat& out);

    // CPU 预处理回退
    int preprocess_cpu(const uint8_t* rgba_data, int src_w, int src_h,
                       int stride, ncnn::Mat& out);

    // ncnn Vulkan 设备
    ncnn::VulkanDevice* vkdev_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device_ = VK_NULL_HANDLE;

    // Vulkan 函数指针 (外部内存扩展)
    PFN_vkGetAndroidHardwareBufferPropertiesANDROID vkGetAHBProperties_ = nullptr;
    PFN_vkGetPhysicalDeviceFormatProperties2 vkGetPhysicalDeviceFormatProperties2_ = nullptr;

    // AHB 缓存：避免每帧重新导入
    std::unordered_map<AHardwareBuffer*, VkAHBImage> ahb_cache_;

    // 预处理参数
    int target_w_ = 640;
    int target_h_ = 640;
    bool vulkan_external_memory_available_ = false;
    bool initialized_ = false;

    // 性能计时
    float capture_time_ms_ = 0.0f;
    float preprocess_time_ms_ = 0.0f;

    // 预处理用的 ncnn Pipeline
    ncnn::Pipeline* preprocess_pipeline_ = nullptr;
    ncnn::VkAllocator* blob_vkallocator_ = nullptr;
    ncnn::VkAllocator* staging_vkallocator_ = nullptr;
};

} // namespace aimassist
