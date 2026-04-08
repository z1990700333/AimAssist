cmake_minimum_required(VERSION 3.22.1)
project(aimassist)

# ncnn 预编译库路径
# 用户需下载 ncnn-YYYYMMDD-android-vulkan.zip 并解压到此目录
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20260113-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(aimassist SHARED
    aimassist_jni.cpp
    ncnn_detector.cpp
    zero_copy_pipeline.cpp
    uinput_injector.cpp
)

target_link_libraries(aimassist
    ncnn
    android
    log
    jnigraphics
    mediandk
    nativewindow
)

# Android 15+ 16KB 页面对齐
target_link_options(aimassist PRIVATE "-Wl,-z,max-page-size=16384")

# 性能优化
target_compile_options(aimassist PRIVATE -O3 -ffast-math -Wall)
