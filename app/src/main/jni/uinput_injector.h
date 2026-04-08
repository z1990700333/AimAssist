#pragma once

#include <linux/uinput.h>

namespace aimassist {

/**
 * uinput 虚拟触摸屏注入器
 * 通过 /dev/uinput 创建独立虚拟触摸设备
 * 脚本点击与真实触摸互不干扰（独立 slot index）
 */
class UinputInjector {
public:
    UinputInjector();
    ~UinputInjector();

    /**
     * 创建虚拟触摸屏设备
     * @param screen_w 屏幕宽度（像素）
     * @param screen_h 屏幕高度（像素）
     * @return 成功返回 fd (>0)，失败返回 -1
     */
    int create(int screen_w, int screen_h);

    /**
     * 注入单次点击（down + up）
     * @param x 点击 X 坐标
     * @param y 点击 Y 坐标
     * @return 成功返回 0
     */
    int tap(int x, int y);

    /**
     * 注入按下事件
     * @param slot 触摸槽位 (0-9)
     * @param tracking_id 跟踪 ID
     * @param x X 坐标
     * @param y Y 坐标
     * @return 成功返回 0
     */
    int touch_down(int slot, int tracking_id, int x, int y);

    /**
     * 注入抬起事件
     * @param slot 触摸槽位
     * @return 成功返回 0
     */
    int touch_up(int slot);

    /**
     * 销毁虚拟设备
     */
    void destroy();

    bool is_created() const { return fd_ >= 0; }
    int get_fd() const { return fd_; }

private:
    void emit(int type, int code, int value);

    int fd_ = -1;
    int screen_w_ = 0;
    int screen_h_ = 0;
    int next_tracking_id_ = 1;
};

} // namespace aimassist
