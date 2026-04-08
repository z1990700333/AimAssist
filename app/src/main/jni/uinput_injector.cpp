#include "uinput_injector.h"
#include <android/log.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cerrno>

#define TAG "UinputInjector"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace aimassist {

UinputInjector::UinputInjector() = default;

UinputInjector::~UinputInjector() {
    destroy();
}

int UinputInjector::create(int screen_w, int screen_h) {
    destroy(); // 清理旧设备

    screen_w_ = screen_w;
    screen_h_ = screen_h;

    fd_ = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if (fd_ < 0) {
        LOGE("打开 /dev/uinput 失败: %s (需要 Root 权限)", strerror(errno));
        return -1;
    }

    // 启用事件类型
    if (ioctl(fd_, UI_SET_EVBIT, EV_ABS) < 0 ||
        ioctl(fd_, UI_SET_EVBIT, EV_KEY) < 0 ||
        ioctl(fd_, UI_SET_EVBIT, EV_SYN) < 0) {
        LOGE("设置事件类型失败: %s", strerror(errno));
        close(fd_);
        fd_ = -1;
        return -2;
    }

    // 启用按键
    ioctl(fd_, UI_SET_KEYBIT, BTN_TOUCH);
    ioctl(fd_, UI_SET_KEYBIT, BTN_TOOL_FINGER);

    // 启用绝对轴 (Multi-touch Protocol B)
    ioctl(fd_, UI_SET_ABSBIT, ABS_MT_SLOT);
    ioctl(fd_, UI_SET_ABSBIT, ABS_MT_TRACKING_ID);
    ioctl(fd_, UI_SET_ABSBIT, ABS_MT_POSITION_X);
    ioctl(fd_, UI_SET_ABSBIT, ABS_MT_POSITION_Y);
    ioctl(fd_, UI_SET_ABSBIT, ABS_MT_PRESSURE);
    ioctl(fd_, UI_SET_ABSBIT, ABS_MT_TOUCH_MAJOR);
    ioctl(fd_, UI_SET_ABSBIT, ABS_X);
    ioctl(fd_, UI_SET_ABSBIT, ABS_Y);

    // 配置绝对轴范围
    auto set_abs = [this](int code, int max_val, int resolution = 0) {
        struct uinput_abs_setup abs = {};
        abs.code = code;
        abs.absinfo.minimum = 0;
        abs.absinfo.maximum = max_val;
        abs.absinfo.resolution = resolution;
        ioctl(fd_, UI_ABS_SETUP, &abs);
    };

    set_abs(ABS_MT_SLOT, 9);                                    // 10 个触摸点
    set_abs(ABS_MT_TRACKING_ID, 65535);
    set_abs(ABS_MT_POSITION_X, screen_w - 1, screen_w / 100);
    set_abs(ABS_MT_POSITION_Y, screen_h - 1, screen_h / 100);
    set_abs(ABS_MT_PRESSURE, 255);
    set_abs(ABS_MT_TOUCH_MAJOR, 255);
    set_abs(ABS_X, screen_w - 1);
    set_abs(ABS_Y, screen_h - 1);

    // 设备描述
    struct uinput_setup usetup = {};
    usetup.id.bustype = BUS_VIRTUAL;
    usetup.id.vendor = 0xAA01;
    usetup.id.product = 0xBB02;
    usetup.id.version = 1;
    strncpy(usetup.name, "AimAssist Virtual Touch", UINPUT_MAX_NAME_SIZE - 1);

    if (ioctl(fd_, UI_DEV_SETUP, &usetup) < 0) {
        LOGE("UI_DEV_SETUP 失败: %s", strerror(errno));
        close(fd_);
        fd_ = -1;
        return -3;
    }

    if (ioctl(fd_, UI_DEV_CREATE) < 0) {
        LOGE("UI_DEV_CREATE 失败: %s", strerror(errno));
        close(fd_);
        fd_ = -1;
        return -4;
    }

    // 等待设备注册
    usleep(100000); // 100ms

    LOGI("虚拟触摸屏创建成功: %dx%d, fd=%d", screen_w, screen_h, fd_);
    return fd_;
}

void UinputInjector::emit(int type, int code, int value) {
    if (fd_ < 0) return;

    struct input_event ev = {};
    ev.type = type;
    ev.code = code;
    ev.value = value;
    // 时间戳由内核填充
    write(fd_, &ev, sizeof(ev));
}

int UinputInjector::tap(int x, int y) {
    if (fd_ < 0) return -1;

    // 边界检查
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= screen_w_) x = screen_w_ - 1;
    if (y >= screen_h_) y = screen_h_ - 1;

    int tracking_id = next_tracking_id_++;
    if (next_tracking_id_ > 65535) next_tracking_id_ = 1;

    // Touch Down
    emit(EV_ABS, ABS_MT_SLOT, 0);
    emit(EV_ABS, ABS_MT_TRACKING_ID, tracking_id);
    emit(EV_ABS, ABS_MT_POSITION_X, x);
    emit(EV_ABS, ABS_MT_POSITION_Y, y);
    emit(EV_ABS, ABS_MT_PRESSURE, 128);
    emit(EV_ABS, ABS_MT_TOUCH_MAJOR, 6);
    emit(EV_ABS, ABS_X, x);
    emit(EV_ABS, ABS_Y, y);
    emit(EV_KEY, BTN_TOUCH, 1);
    emit(EV_KEY, BTN_TOOL_FINGER, 1);
    emit(EV_SYN, SYN_REPORT, 0);

    // Touch Up (极短延迟)
    usleep(1000); // 1ms

    emit(EV_ABS, ABS_MT_SLOT, 0);
    emit(EV_ABS, ABS_MT_TRACKING_ID, -1); // -1 = 抬起
    emit(EV_KEY, BTN_TOUCH, 0);
    emit(EV_KEY, BTN_TOOL_FINGER, 0);
    emit(EV_SYN, SYN_REPORT, 0);

    return 0;
}

int UinputInjector::touch_down(int slot, int tracking_id, int x, int y) {
    if (fd_ < 0) return -1;

    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= screen_w_) x = screen_w_ - 1;
    if (y >= screen_h_) y = screen_h_ - 1;
    if (slot < 0 || slot > 9) return -2;

    emit(EV_ABS, ABS_MT_SLOT, slot);
    emit(EV_ABS, ABS_MT_TRACKING_ID, tracking_id);
    emit(EV_ABS, ABS_MT_POSITION_X, x);
    emit(EV_ABS, ABS_MT_POSITION_Y, y);
    emit(EV_ABS, ABS_MT_PRESSURE, 128);
    emit(EV_ABS, ABS_MT_TOUCH_MAJOR, 6);

    if (slot == 0) {
        emit(EV_ABS, ABS_X, x);
        emit(EV_ABS, ABS_Y, y);
        emit(EV_KEY, BTN_TOUCH, 1);
        emit(EV_KEY, BTN_TOOL_FINGER, 1);
    }

    emit(EV_SYN, SYN_REPORT, 0);
    return 0;
}

int UinputInjector::touch_up(int slot) {
    if (fd_ < 0) return -1;
    if (slot < 0 || slot > 9) return -2;

    emit(EV_ABS, ABS_MT_SLOT, slot);
    emit(EV_ABS, ABS_MT_TRACKING_ID, -1);

    if (slot == 0) {
        emit(EV_KEY, BTN_TOUCH, 0);
        emit(EV_KEY, BTN_TOOL_FINGER, 0);
    }

    emit(EV_SYN, SYN_REPORT, 0);
    return 0;
}

void UinputInjector::destroy() {
    if (fd_ >= 0) {
        ioctl(fd_, UI_DEV_DESTROY);
        close(fd_);
        fd_ = -1;
        LOGI("虚拟触摸屏已销毁");
    }
}

} // namespace aimassist
