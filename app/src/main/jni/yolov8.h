// YOLOv8 检测器 - 参考 nihui/ncnn-android-yolov8
// 去掉摄像头部分，仅保留检测核心 + 屏幕截图输入

#ifndef YOLOV8_H
#define YOLOV8_H

#include <vector>
#include <string>
#include <net.h>

namespace aimassist {

struct Object
{
    float x;      // bbox left (模型坐标)
    float y;      // bbox top
    float w;      // bbox width
    float h;      // bbox height
    int label;
    float prob;
};

class YOLOv8
{
public:
    YOLOv8();
    ~YOLOv8();

    // 从文件系统路径加载
    int load(const char* parampath, const char* modelpath, bool use_gpu = false);

    // 从 RGBA 像素数据检测 (屏幕截图输入)
    int detect_from_rgba(const unsigned char* rgba, int img_w, int img_h, int stride,
                         std::vector<Object>& objects,
                         float conf_threshold = 0.25f, float nms_threshold = 0.45f);

    void set_target_size(int target_size);
    int get_target_size() const { return target_size; }
    float get_inference_time_ms() const { return inference_time_ms; }
    bool is_loaded() const { return loaded; }

private:
    // YOLOv8 后处理 (参考 nihui 实现)
    static void generate_proposals(const ncnn::Mat& feat_blob, int stride,
                                   const ncnn::Mat& in_pad, int target_size,
                                   float prob_threshold,
                                   std::vector<Object>& objects, int num_class);

    static void qsort_descent_inplace(std::vector<Object>& objects);
    static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right);
    static float intersection_area(const Object& a, const Object& b);
    static void nms_sorted_bboxes(const std::vector<Object>& objects,
                                  std::vector<int>& picked, float nms_threshold);

    ncnn::Net net;
    int target_size;
    bool loaded;
    bool use_gpu_;
    float inference_time_ms;
};

} // namespace aimassist

#endif // YOLOV8_H
