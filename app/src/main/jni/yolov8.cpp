// YOLOv8 检测器实现 - 参考 nihui/ncnn-android-yolov8 的 yolov8_det.cpp
// 去掉 OpenCV 依赖，去掉摄像头，仅保留屏幕截图 RGBA 输入

#include "yolov8.h"
#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>

#if __ARM_NEON
#include <arm_neon.h>
#endif

#include <benchmark.h>

#define TAG "YOLOv8"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace aimassist {

// softmax 用于 DFL
static inline float softmax_sum(const float* src, float* dst, int length)
{
    float alpha = -FLT_MAX;
    for (int c = 0; c < length; c++)
        alpha = std::max(alpha, src[c]);

    float denominator = 0;
    for (int c = 0; c < length; c++)
    {
        dst[c] = std::exp(src[c] - alpha);
        denominator += dst[c];
    }

    for (int c = 0; c < length; c++)
        dst[c] /= denominator;

    float sum = 0;
    for (int c = 0; c < length; c++)
        sum += dst[c] * c;

    return sum;
}

YOLOv8::YOLOv8()
    : target_size(320), loaded(false), use_gpu_(false), inference_time_ms(0)
{
}

YOLOv8::~YOLOv8()
{
    net.clear();
}

int YOLOv8::load(const char* parampath, const char* modelpath, bool use_gpu)
{
    net.clear();
    loaded = false;
    use_gpu_ = use_gpu;

    net.opt = ncnn::Option();

#if NCNN_VULKAN
    if (use_gpu && ncnn::get_gpu_count() > 0)
    {
        net.opt.use_vulkan_compute = true;
        net.opt.use_fp16_packed = true;
        net.opt.use_fp16_storage = true;
        net.opt.use_fp16_arithmetic = false;
        LOGI("Vulkan GPU 推理已启用");
    }
    else
    {
        net.opt.use_vulkan_compute = false;
        if (use_gpu)
            LOGI("请求 GPU 但无 Vulkan 设备，回退 CPU");
    }
#else
    net.opt.use_vulkan_compute = false;
#endif

    net.opt.num_threads = 4;

    int ret = net.load_param(parampath);
    if (ret != 0)
    {
        LOGE("加载 param 失败: %s (ret=%d)", parampath, ret);
        return -1;
    }

    ret = net.load_model(modelpath);
    if (ret != 0)
    {
        LOGE("加载 bin 失败: %s (ret=%d)", modelpath, ret);
        return -2;
    }

    loaded = true;
    LOGI("模型加载成功: param=%s, gpu=%d, target_size=%d", parampath, use_gpu, target_size);
    return 0;
}

void YOLOv8::set_target_size(int ts)
{
    target_size = ts;
}

// ============================================================================
// 从 RGBA 屏幕截图检测
// ============================================================================

int YOLOv8::detect_from_rgba(const unsigned char* rgba, int img_w, int img_h, int stride,
                              std::vector<Object>& objects,
                              float conf_threshold, float nms_threshold)
{
    if (!loaded) return -1;

    auto t0 = std::chrono::high_resolution_clock::now();

    // letterbox resize，保持宽高比
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = (int)(h * scale);
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = (int)(w * scale);
    }

    // RGBA → RGB + resize
    ncnn::Mat in;
    if ((int)(stride / 4) == img_w)
    {
        // 无 padding
        in = ncnn::Mat::from_pixels_resize(rgba, ncnn::Mat::PIXEL_RGBA2RGB,
                                           img_w, img_h, w, h);
    }
    else
    {
        // 有 stride padding，逐行拷贝
        std::vector<unsigned char> clean(img_w * img_h * 4);
        for (int y = 0; y < img_h; y++)
            memcpy(clean.data() + y * img_w * 4, rgba + y * stride, img_w * 4);
        in = ncnn::Mat::from_pixels_resize(clean.data(), ncnn::Mat::PIXEL_RGBA2RGB,
                                           img_w, img_h, w, h);
    }

    // letterbox pad to target_size x target_size
    int wpad = target_size - w;
    int hpad = target_size - h;
    ncnn::Mat in_pad;
    if (wpad > 0 || hpad > 0)
    {
        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2,
                               wpad / 2, wpad - wpad / 2,
                               ncnn::BORDER_CONSTANT, 114.f);
    }
    else
    {
        in_pad = in;
    }

    // normalize [0,255] → [0,1]
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    // 推理
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in_pad);

    // 多尺度输出 (stride 8, 16, 32)
    std::vector<Object> proposals;

    // 自动探测输出层
    // 尝试 out0, out1, out2 (标准 YOLOv8 三头输出)
    {
        ncnn::Mat out;
        if (ex.extract("out0", out) == 0)
        {
            // 检查是否是单输出 (合并后的)
            // 单输出: out0.w = num_detections, out0.h = 4+nc 或反过来
            if (out.h > 6 && out.w > 6)
            {
                // 可能是合并输出，按 YOLOv8 单输出处理
                int num_class = 0;
                if (out.w > out.h)
                {
                    // [4+nc, num_det] 格式
                    num_class = out.h - 4;
                }
                else
                {
                    // [num_det, 4+nc] 格式 — 需要转置
                    num_class = out.w - 4;
                }

                if (num_class > 0 && num_class < 1000)
                {
                    // 单输出模式
                    generate_proposals(out, 0, in_pad, target_size,
                                       conf_threshold, proposals, num_class);
                }
            }
            else
            {
                // 多头输出 stride=8
                ncnn::Mat out1, out2;
                int num_class = out.h - 4;
                if (num_class <= 0) num_class = out.w - 4;
                if (num_class <= 0) num_class = 80;

                generate_proposals(out, 8, in_pad, target_size,
                                   conf_threshold, proposals, num_class);

                if (ex.extract("out1", out1) == 0)
                    generate_proposals(out1, 16, in_pad, target_size,
                                       conf_threshold, proposals, num_class);

                if (ex.extract("out2", out2) == 0)
                    generate_proposals(out2, 32, in_pad, target_size,
                                       conf_threshold, proposals, num_class);
            }
        }
    }

    // NMS
    qsort_descent_inplace(proposals);
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = (int)picked.size();
    objects.resize(count);

    // 还原到原始图像坐标
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // letterbox 偏移
        float x0 = (objects[i].x - (wpad / 2)) / scale;
        float y0 = (objects[i].y - (hpad / 2)) / scale;
        float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
        float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].x = x0;
        objects[i].y = y0;
        objects[i].w = x1 - x0;
        objects[i].h = y1 - y0;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    inference_time_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    return 0;
}

// ============================================================================
// YOLOv8 后处理 - 参考 nihui 的 generate_proposals
// ============================================================================

void YOLOv8::generate_proposals(const ncnn::Mat& feat_blob, int stride,
                                 const ncnn::Mat& in_pad, int target_size,
                                 float prob_threshold,
                                 std::vector<Object>& objects, int num_class)
{
    if (stride == 0)
    {
        // 单输出模式: feat_blob 是合并后的所有检测
        // 格式: [4+nc, num_det] 或 [num_det, 4+nc]
        const int feat_h = feat_blob.h;
        const int feat_w = feat_blob.w;

        bool transposed = (feat_w > feat_h); // [4+nc, num_det] → w=num_det
        int num_det = transposed ? feat_w : feat_h;
        int row_size = transposed ? feat_h : feat_w;
        int nc = row_size - 4;
        if (nc != num_class) nc = num_class;

        for (int i = 0; i < num_det; i++)
        {
            // 读取 bbox
            float cx, cy, bw, bh;
            if (transposed)
            {
                cx = feat_blob.row(0)[i];
                cy = feat_blob.row(1)[i];
                bw = feat_blob.row(2)[i];
                bh = feat_blob.row(3)[i];
            }
            else
            {
                const float* row = feat_blob.row(i);
                cx = row[0];
                cy = row[1];
                bw = row[2];
                bh = row[3];
            }

            // 找最大类别分数
            int class_index = 0;
            float class_score = -FLT_MAX;
            for (int c = 0; c < nc; c++)
            {
                float score;
                if (transposed)
                    score = feat_blob.row(4 + c)[i];
                else
                    score = feat_blob.row(i)[4 + c];

                if (score > class_score)
                {
                    class_index = c;
                    class_score = score;
                }
            }

            if (class_score < prob_threshold)
                continue;

            Object obj;
            obj.x = cx - bw * 0.5f;
            obj.y = cy - bh * 0.5f;
            obj.w = bw;
            obj.h = bh;
            obj.label = class_index;
            obj.prob = class_score;
            objects.push_back(obj);
        }
        return;
    }

    // 多头输出模式 (stride 8/16/32) — 标准 YOLOv8 DFL 解码
    const int reg_max = 16;
    const int feat_h = feat_blob.h;
    const int feat_w = feat_blob.w;

    // feat_blob: [4*reg_max + num_class, feat_h * feat_w]
    // 或 [feat_h * feat_w, 4*reg_max + num_class]
    bool transposed = (feat_w > feat_h);
    int num_grid;
    int expected_row_size;

    if (transposed)
    {
        num_grid = feat_w;
        expected_row_size = feat_h;
    }
    else
    {
        num_grid = feat_h;
        expected_row_size = feat_w;
    }

    int num_grid_y = target_size / stride;
    int num_grid_x = target_size / stride;

    for (int i = 0; i < num_grid; i++)
    {
        int grid_y = i / num_grid_x;
        int grid_x = i % num_grid_x;

        // 找最大类别分数
        int class_index = 0;
        float class_score = -FLT_MAX;
        for (int c = 0; c < num_class; c++)
        {
            float score;
            if (transposed)
                score = feat_blob.row(4 * reg_max + c)[i];
            else
                score = feat_blob.row(i)[4 * reg_max + c];

            if (score > class_score)
            {
                class_index = c;
                class_score = score;
            }
        }

        if (class_score < prob_threshold)
            continue;

        // DFL 解码 bbox
        float pred_ltrb[4];
        for (int k = 0; k < 4; k++)
        {
            float dis_src[16];
            float dis_after_sm[16];
            for (int l = 0; l < reg_max; l++)
            {
                if (transposed)
                    dis_src[l] = feat_blob.row(k * reg_max + l)[i];
                else
                    dis_src[l] = feat_blob.row(i)[k * reg_max + l];
            }
            pred_ltrb[k] = softmax_sum(dis_src, dis_after_sm, reg_max);
        }

        float pb_cx = (grid_x + 0.5f) * stride;
        float pb_cy = (grid_y + 0.5f) * stride;

        float x0 = pb_cx - pred_ltrb[0] * stride;
        float y0 = pb_cy - pred_ltrb[1] * stride;
        float x1 = pb_cx + pred_ltrb[2] * stride;
        float y1 = pb_cy + pred_ltrb[3] * stride;

        Object obj;
        obj.x = x0;
        obj.y = y0;
        obj.w = x1 - x0;
        obj.h = y1 - y0;
        obj.label = class_index;
        obj.prob = class_score;
        objects.push_back(obj);
    }
}

// ============================================================================
// 排序 + NMS (参考 nihui)
// ============================================================================

float YOLOv8::intersection_area(const Object& a, const Object& b)
{
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.w, b.x + b.w);
    float y2 = std::min(a.y + a.h, b.y + b.h);
    float w = std::max(0.f, x2 - x1);
    float h = std::max(0.f, y2 - y1);
    return w * h;
}

void YOLOv8::qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p) i++;
        while (objects[j].prob < p) j--;
        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    if (left < j) qsort_descent_inplace(objects, left, j);
    if (i < right) qsort_descent_inplace(objects, i, right);
}

void YOLOv8::qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty()) return;
    qsort_descent_inplace(objects, 0, (int)objects.size() - 1);
}

void YOLOv8::nms_sorted_bboxes(const std::vector<Object>& objects,
                                 std::vector<int>& picked, float nms_threshold)
{
    picked.clear();
    const int n = (int)objects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
        areas[i] = objects[i].w * objects[i].h;

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];
        bool keep = true;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];
            float inter = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter;
            if (inter / union_area > nms_threshold)
            {
                keep = false;
                break;
            }
        }
        if (keep) picked.push_back(i);
    }
}

} // namespace aimassist
