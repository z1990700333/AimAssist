#pragma once

#include <string>
#include <vector>
#include <net.h>
#include <gpu.h>

namespace aimassist {

struct Object {
    float x;      // bbox left
    float y;      // bbox top
    float w;      // bbox width
    float h;      // bbox height
    int label;
    float prob;
};

class NcnnDetector {
public:
    NcnnDetector();
    ~NcnnDetector();

    /**
     * 从文件系统路径加载模型
     * @param param_path .param 文件路径
     * @param bin_path .bin 文件路径
     * @param use_gpu 是否使用 Vulkan GPU
     * @return 成功返回 0
     */
    int load(const char* param_path, const char* bin_path, bool use_gpu);

    /**
     * 执行目标检测
     * @param in 输入 ncnn::Mat (RGB, 已归一化)
     * @param objects 输出检测结果
     * @param conf_threshold 置信度阈值
     * @param nms_threshold NMS 阈值
     * @return 成功返回 0
     */
    int detect(const ncnn::Mat& in, std::vector<Object>& objects,
               float conf_threshold = 0.5f, float nms_threshold = 0.45f);

    /**
     * 使用 VkMat 执行检测 (零拷贝路径)
     */
    int detect_vulkan(const ncnn::VkMat& in, std::vector<Object>& objects,
                      float conf_threshold = 0.5f, float nms_threshold = 0.45f);

    /**
     * 获取模型输入尺寸
     */
    int get_input_size() const { return input_size_; }

    /**
     * 获取上次推理耗时 (ms)
     */
    float get_inference_time() const { return inference_time_ms_; }

    /**
     * 释放资源
     */
    void release();

    bool is_loaded() const { return loaded_; }

private:
    void detect_yolov5(const ncnn::Mat& output, std::vector<Object>& objects,
                       float conf_threshold, int img_w, int img_h);
    void detect_yolov8(const ncnn::Mat& output, std::vector<Object>& objects,
                       float conf_threshold, int img_w, int img_h);
    void nms_sorted_bboxes(std::vector<Object>& objects, float nms_threshold);
    float intersection_area(const Object& a, const Object& b);

    ncnn::Net net_;
    bool loaded_ = false;
    bool use_gpu_ = false;
    int input_size_ = 640;
    int num_class_ = 80;
    bool is_yolov8_ = false;  // true=YOLOv8格式, false=YOLOv5格式
    float inference_time_ms_ = 0.0f;

    // 输入/输出 blob 名称
    std::string input_name_;
    std::string output_name_;
};

} // namespace aimassist
