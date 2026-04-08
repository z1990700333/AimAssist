#include "ncnn_detector.h"
#include <android/log.h>
#include <algorithm>
#include <chrono>
#include <cmath>

#define TAG "NcnnDetector"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace aimassist {

NcnnDetector::NcnnDetector() = default;

NcnnDetector::~NcnnDetector() {
    release();
}

int NcnnDetector::load(const char* param_path, const char* bin_path, bool use_gpu) {
    release();

    use_gpu_ = use_gpu;

    // 配置 ncnn 选项
    net_.opt = ncnn::Option();

#if NCNN_VULKAN
    if (use_gpu && ncnn::get_gpu_count() > 0) {
        net_.opt.use_vulkan_compute = true;
        net_.opt.use_fp16_packed = true;
        net_.opt.use_fp16_storage = true;
        net_.opt.use_fp16_arithmetic = false; // 部分 GPU 不支持
        net_.opt.use_int8_packed = true;
        net_.opt.use_int8_storage = true;
        LOGI("Vulkan GPU 推理已启用 (FP16)");
    } else {
        net_.opt.use_vulkan_compute = false;
        LOGI("使用 CPU 推理");
    }
#else
    net_.opt.use_vulkan_compute = false;
    LOGI("ncnn 未编译 Vulkan 支持，使用 CPU");
#endif

    net_.opt.num_threads = 4;
    net_.opt.lightmode = true;

    // 从文件系统加载模型
    int ret = net_.load_param(param_path);
    if (ret != 0) {
        LOGE("加载 param 失败: %s (ret=%d)", param_path, ret);
        return -1;
    }

    ret = net_.load_model(bin_path);
    if (ret != 0) {
        LOGE("加载 bin 失败: %s (ret=%d)", bin_path, ret);
        return -2;
    }

    // 动态解析模型输入/输出
    const std::vector<int>& input_indexes = net_.input_indexes();
    const std::vector<int>& output_indexes = net_.output_indexes();

    if (input_indexes.empty() || output_indexes.empty()) {
        LOGE("模型没有输入或输出层");
        return -3;
    }

    // 获取输入/输出 blob 名称
    const std::vector<const char*>& input_names = net_.input_names();
    const std::vector<const char*>& output_names = net_.output_names();

    input_name_ = input_names[0];
    output_name_ = output_names[0];

    LOGI("模型输入层: %s, 输出层: %s", input_name_.c_str(), output_name_.c_str());

    // 通过试推理获取输入尺寸和输出格式
    // 创建一个小的 extractor 来探测
    {
        ncnn::Extractor ex = net_.create_extractor();

        // 尝试用 1x1 输入探测输入形状
        // ncnn 的 param 文件中包含 Input 层的 w/h 信息
        // 我们通过读取 blob 的 shape 来获取
        const std::vector<ncnn::Blob>& blobs = net_.blobs();
        for (size_t i = 0; i < blobs.size(); i++) {
            if (blobs[i].name == input_name_) {
                // Input blob 的 shape 在 param 中定义
                ncnn::Mat shape = blobs[i].shape;
                if (!shape.empty()) {
                    input_size_ = shape.w > 0 ? shape.w : 640;
                    LOGI("从模型解析输入尺寸: %d", input_size_);
                }
                break;
            }
        }

        // 如果无法从 blob shape 获取，使用默认值
        if (input_size_ <= 0) {
            input_size_ = 640;
            LOGI("使用默认输入尺寸: %d", input_size_);
        }

        // 探测输出格式：YOLOv5 vs YOLOv8
        ncnn::Mat dummy_in(input_size_, input_size_, 3);
        dummy_in.fill(0.0f);
        ex.input(input_name_.c_str(), dummy_in);

        ncnn::Mat dummy_out;
        ex.extract(output_name_.c_str(), dummy_out);

        if (dummy_out.dims == 3) {
            // YOLOv5: [1, num_detections, 5+num_class] → dims=2, w=5+nc, h=num_det
            // YOLOv8: [1, 4+num_class, num_detections] → dims=2, w=num_det, h=4+nc
            // 或 dims=3 的情况
            int dim0 = dummy_out.h;
            int dim1 = dummy_out.w;

            if (dim0 > dim1) {
                // dim0 大 → YOLOv5 格式 (num_det > 5+nc)
                is_yolov8_ = false;
                num_class_ = dim1 - 5;
                LOGI("检测到 YOLOv5 格式: detections=%d, classes=%d", dim0, num_class_);
            } else {
                // dim1 大 → YOLOv8 格式 (num_det > 4+nc)
                is_yolov8_ = true;
                num_class_ = dim0 - 4;
                LOGI("检测到 YOLOv8 格式: detections=%d, classes=%d", dim1, num_class_);
            }
        } else if (dummy_out.dims == 2) {
            int dim0 = dummy_out.h;
            int dim1 = dummy_out.w;

            if (dim1 > dim0) {
                is_yolov8_ = true;
                num_class_ = dim0 - 4;
            } else {
                is_yolov8_ = false;
                num_class_ = dim1 - 5;
            }
            LOGI("输出 dims=2: %s 格式, classes=%d",
                 is_yolov8_ ? "YOLOv8" : "YOLOv5", num_class_);
        }

        if (num_class_ <= 0) {
            LOGE("无法解析模型类别数 (num_class=%d)", num_class_);
            num_class_ = 80; // 回退到 COCO 默认
        }
    }

    loaded_ = true;
    LOGI("模型加载成功: input_size=%d, num_class=%d, format=%s, gpu=%s",
         input_size_, num_class_, is_yolov8_ ? "YOLOv8" : "YOLOv5",
         use_gpu_ ? "true" : "false");

    return 0;
}

int NcnnDetector::detect(const ncnn::Mat& in, std::vector<Object>& objects,
                         float conf_threshold, float nms_threshold) {
    if (!loaded_) return -1;

    auto t0 = std::chrono::high_resolution_clock::now();

    ncnn::Extractor ex = net_.create_extractor();
    ex.input(input_name_.c_str(), in);

    ncnn::Mat output;
    ex.extract(output_name_.c_str(), output);

    auto t1 = std::chrono::high_resolution_clock::now();
    inference_time_ms_ = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // 后处理
    objects.clear();
    if (is_yolov8_) {
        detect_yolov8(output, objects, conf_threshold, input_size_, input_size_);
    } else {
        detect_yolov5(output, objects, conf_threshold, input_size_, input_size_);
    }

    // NMS
    nms_sorted_bboxes(objects, nms_threshold);

    return 0;
}

int NcnnDetector::detect_vulkan(const ncnn::VkMat& in, std::vector<Object>& objects,
                                float conf_threshold, float nms_threshold) {
    if (!loaded_) return -1;

#if NCNN_VULKAN
    auto t0 = std::chrono::high_resolution_clock::now();

    ncnn::Extractor ex = net_.create_extractor();

    // VkMat 输入路径 - 全程 GPU
    ncnn::VkCompute cmd(net_.vulkan_device());
    ex.input(input_name_.c_str(), in);

    ncnn::Mat output;
    ex.extract(output_name_.c_str(), output);

    auto t1 = std::chrono::high_resolution_clock::now();
    inference_time_ms_ = std::chrono::duration<float, std::milli>(t1 - t0).count();

    objects.clear();
    if (is_yolov8_) {
        detect_yolov8(output, objects, conf_threshold, input_size_, input_size_);
    } else {
        detect_yolov5(output, objects, conf_threshold, input_size_, input_size_);
    }

    nms_sorted_bboxes(objects, nms_threshold);

    return 0;
#else
    // 无 Vulkan 支持，回退到 CPU
    ncnn::Mat cpu_in;
    // VkMat 无法直接转换，需要通过其他方式
    return -1;
#endif
}

void NcnnDetector::detect_yolov5(const ncnn::Mat& output, std::vector<Object>& objects,
                                  float conf_threshold, int img_w, int img_h) {
    // YOLOv5 输出格式: [num_detections, 5 + num_class]
    // 每行: [cx, cy, w, h, obj_conf, class_0_conf, class_1_conf, ...]
    const int num_detections = output.h;
    const int num_cols = output.w;

    for (int i = 0; i < num_detections; i++) {
        const float* row = output.row(i);

        float obj_conf = row[4];
        if (obj_conf < conf_threshold) continue;

        // 找最大类别分数
        int best_class = 0;
        float best_score = 0.0f;
        for (int c = 0; c < num_class_; c++) {
            float score = row[5 + c];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        float final_score = obj_conf * best_score;
        if (final_score < conf_threshold) continue;

        // 解码 bbox (cx, cy, w, h → x, y, w, h)
        float cx = row[0];
        float cy = row[1];
        float bw = row[2];
        float bh = row[3];

        Object obj;
        obj.x = cx - bw * 0.5f;
        obj.y = cy - bh * 0.5f;
        obj.w = bw;
        obj.h = bh;
        obj.label = best_class;
        obj.prob = final_score;

        objects.push_back(obj);
    }
}

void NcnnDetector::detect_yolov8(const ncnn::Mat& output, std::vector<Object>& objects,
                                  float conf_threshold, int img_w, int img_h) {
    // YOLOv8 输出格式: [4 + num_class, num_detections] (转置)
    // 需要按列读取
    const int num_rows = output.h;  // 4 + num_class
    const int num_cols = output.w;  // num_detections

    // 转置后处理
    for (int j = 0; j < num_cols; j++) {
        // 读取 bbox
        float cx = output.row(0)[j];
        float cy = output.row(1)[j];
        float bw = output.row(2)[j];
        float bh = output.row(3)[j];

        // 找最大类别分数 (YOLOv8 没有 obj_conf，直接是类别分数)
        int best_class = 0;
        float best_score = 0.0f;
        for (int c = 0; c < num_class_; c++) {
            float score = output.row(4 + c)[j];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }

        if (best_score < conf_threshold) continue;

        Object obj;
        obj.x = cx - bw * 0.5f;
        obj.y = cy - bh * 0.5f;
        obj.w = bw;
        obj.h = bh;
        obj.label = best_class;
        obj.prob = best_score;

        objects.push_back(obj);
    }
}

float NcnnDetector::intersection_area(const Object& a, const Object& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    return w * h;
}

void NcnnDetector::nms_sorted_bboxes(std::vector<Object>& objects, float nms_threshold) {
    if (objects.empty()) return;

    // 按置信度降序排序
    std::sort(objects.begin(), objects.end(),
              [](const Object& a, const Object& b) { return a.prob > b.prob; });

    std::vector<Object> result;
    std::vector<bool> suppressed(objects.size(), false);

    for (size_t i = 0; i < objects.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(objects[i]);

        float area_i = objects[i].w * objects[i].h;

        for (size_t j = i + 1; j < objects.size(); j++) {
            if (suppressed[j]) continue;

            float inter = intersection_area(objects[i], objects[j]);
            float area_j = objects[j].w * objects[j].h;
            float iou = inter / (area_i + area_j - inter);

            if (iou > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }

    objects = std::move(result);
}

void NcnnDetector::release() {
    if (loaded_) {
        net_.clear();
        loaded_ = false;
        LOGI("检测器已释放");
    }
}

} // namespace aimassist
