# AimAssist - 安卓端极致低延迟 AI 视觉推理系统

基于 **ncnn Vulkan** 的实时画面识别与自动化点击系统，全链路延迟 < 25ms。

## 技术架构

```
MediaProjection → AHardwareBuffer → VkImage → ncnn Vulkan 推理 → 自动点击
                     零拷贝 GPU 管线，严禁 CPU 缓存
```

### 核心特性

- **ncnn Vulkan 后端**：FP16 推理，算子融合，全 GPU 计算
- **零拷贝管线**：AHardwareBuffer → Vulkan 外部内存 → ncnn VkMat
- **双模式点击**：
  - 非 Root：AccessibilityService dispatchGesture (1ms duration)
  - Root：/dev/uinput 虚拟触摸屏（独立 slot，不干扰真实触摸）
- **动态模型加载**：支持用户自定义 .param + .bin 模型
- **自动格式检测**：自动识别 YOLOv5/v8 输出格式
- **Android 14-16 适配**：16KB 页面对齐，SELinux 策略注入

### 性能指标 (骁龙 8 Gen 2)

| 阶段 | 目标延迟 |
|------|---------|
| 截图 + 纹理映射 | < 2ms |
| GPU 预处理 | < 1ms |
| ncnn 推理 (YOLOv8n 320) | < 15ms |
| uinput 事件注入 | < 1ms |
| **总链路 (P99)** | **< 25ms** |

## 编译要求

### 环境

- Android Studio Ladybug 或更高版本
- NDK 27.2.12479018
- CMake 3.22.1
- Gradle 8.2+
- JDK 17

### ncnn 预编译库

1. 从 [ncnn Releases](https://github.com/Tencent/ncnn/releases) 下载 `ncnn-YYYYMMDD-android-vulkan.zip`
2. 解压到 `app/src/main/jni/ncnn-20260113-android-vulkan/`

目录结构：
```
app/src/main/jni/
├── ncnn-20260113-android-vulkan/
│   └── arm64-v8a/
│       ├── include/
│       └── lib/
│           └── cmake/ncnn/
├── CMakeLists.txt
├── aimassist_jni.cpp
├── ncnn_detector.h/cpp
├── zero_copy_pipeline.h/cpp
└── uinput_injector.h/cpp
```

### 编译

```bash
./gradlew assembleDebug
```

### 验证 16KB 页面对齐

```bash
readelf -l app/build/intermediates/merged_native_libs/debug/out/lib/arm64-v8a/libaimassist.so | grep LOAD
# 所有 LOAD 段 Align 应为 0x4000
```

## 使用说明

### 1. 准备模型

将 YOLO 模型转换为 ncnn 格式：

```bash
# PyTorch → ONNX → ncnn
python export.py --weights yolov8n.pt --include onnx
./onnx2ncnn yolov8n.onnx yolov8n.param yolov8n.bin

# 启用 FP16 (推荐)
./ncnnoptimize yolov8n.param yolov8n.bin yolov8n-fp16.param yolov8n-fp16.bin 1
```

### 2. 加载模型

1. 打开 AimAssist → 设置
2. 选择 `.param` 参数文件
3. 选择 `.bin` 权重文件
4. 调整置信度阈值、NMS 阈值、输入尺寸

### 3. 启动检测

1. 返回主页 → 点击「开始检测」
2. 授权屏幕录制权限
3. 实时查看 FPS、检测数、推理耗时

### 4. 自动点击

**非 Root 模式：**
- 开启「自动点击」开关
- 前往系统设置启用 AimAssist 无障碍服务

**Root 模式：**
- 设置中选择「Root uinput」点击模式
- 需要 Magisk/KernelSU Root 权限
- SELinux 策略自动注入

### 参数说明

| 参数 | 说明 | 建议值 |
|------|------|--------|
| 置信度阈值 | 只显示置信度高于此值的目标 | 0.3-0.5 |
| NMS 阈值 | 非极大值抑制，去除重叠框 | 0.45 |
| 输入尺寸 | 模型输入分辨率 | 320(快)/640(准) |
| 点击延迟 | 检测到目标后延迟点击(ms) | 0 |
| 点击偏移 | 相对于检测框中心的偏移 | 0,0 |
| 目标类别 | 只点击特定类别，留空检测所有 | - |

## 项目结构

```
app/src/main/
├── jni/                          # C++ 原生层
│   ├── CMakeLists.txt            # 构建配置
│   ├── aimassist_jni.cpp         # JNI 桥接
│   ├── ncnn_detector.h/cpp       # ncnn 推理引擎
│   ├── zero_copy_pipeline.h/cpp  # 零拷贝 GPU 管线
│   └── uinput_injector.h/cpp     # 虚拟触摸屏
├── java/com/aimassist/app/
│   ├── ncnn/NcnnDetector.kt      # JNI 封装
│   ├── service/
│   │   ├── DetectionService.kt   # 检测服务
│   │   └── AutoClickService.kt   # 无障碍点击
│   ├── data/DetectionResult.kt   # 数据模型
│   ├── utils/SettingsManager.kt  # 设置管理
│   ├── MainActivity.kt           # 主界面
│   ├── SettingsActivity.kt       # 设置界面
│   └── App.kt                    # 应用入口
└── res/                          # UI 资源 (Material Design 3 深色主题)
```

## 技术细节

### 零拷贝管线

```
MediaProjection
  └→ VirtualDisplay → ImageReader (HardwareBuffer)
       └→ AHardwareBuffer
            └→ VK_ANDROID_external_memory_android_hardware_buffer
                 └→ VkImage (GPU 内存，无 CPU 拷贝)
                      └→ Vulkan Compute 预处理 (RGBA→RGB + resize + normalize)
                           └→ ncnn::VkMat → Vulkan 推理
```

### uinput 虚拟触摸屏

通过 `/dev/uinput` 创建独立虚拟触摸设备，使用 Multi-touch Protocol B：
- 独立 slot index，系统将脚本点击和真实触摸视为两个物理源
- 脚本点击时用户手指不需要松开，彻底解决断触问题

### Android 15+ 适配

- 所有 `.so` 使用 `-Wl,-z,max-page-size=16384` 链接
- CMake 参数：`-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON`

## 技术栈

- **Kotlin** - Android 应用层
- **C++17** - 原生推理层
- **ncnn** - 高性能推理框架 (Vulkan 后端)
- **Vulkan** - GPU 计算与零拷贝管线
- **Material Design 3** - 现代化深色 UI
- **MediaProjection** - 屏幕捕获
- **AccessibilityService** - 无障碍点击
- **uinput** - Linux 内核级触摸注入

## 许可证

MIT License

## 致谢

- [Tencent/ncnn](https://github.com/Tencent/ncnn) - 高性能推理框架
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 目标检测模型
- [Material Components](https://github.com/material-components/material-components-android) - UI 组件
