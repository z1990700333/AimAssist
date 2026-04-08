#pragma once

#include <android/hardware_buffer.h>

// ncnn 头文件 (内含 simplevk.h 提供 Vulkan 类型定义)
// 注意：不可同时包含系统 <vulkan/vulkan.h>，否则会导致类型重定义
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
     * 处理一帧：自动选择最优路径
     * 返回 ncnn::Mat
     */
    int process(AHardwareBuffer* ahb, int src_w, int src_h, ncnn::Mat& out);

    /**
     * 获取各阶段耗时
     */
    float get_capture_time_ms() const { return capture_time_ms_; }
    float get_preprocess_time_ms() const { return preprocess_time_ms_; }

    void release();

private:
    // CPU 预处理回退
    int preprocess_cpu(const uint8_t* rgba_data, int src_w, int src_h,
                       int stride, ncnn::Mat& out);

    // ncnn Vulkan 设备
    ncnn::VulkanDevice* vkdev_ = nullptr;

    // 预处理参数
    int target_w_ = 640;
    int target_h_ = 640;
    bool initialized_ = false;

    // 性能计时
    float capture_time_ms_ = 0.0f;
    float preprocess_time_ms_ = 0.0f;

    // ncnn VkAllocator
    ncnn::VkAllocator* blob_vkallocator_ = nullptr;
    ncnn::VkAllocator* staging_vkallocator_ = nullptr;
};

} // namespace aimassist
