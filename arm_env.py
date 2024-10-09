"""
Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
"""

# 开源数值计算库
import numpy as np
# 跨平台窗口和多媒体库，图形用户界面
import pyglet


#pyglet.clock.set_fps_limit(10000)

# 定义一个机械臂类，用于控制和观察机械臂的行为
class ArmEnv(object):
    # 边界
    action_bound = [-1, 1]
    # 动作的维度。两个不同的控制输入
    action_dim = 2
    # 状态空间的维度。位置，速度，加速度
    state_dim = 7
    # 系统状态更新的频率
    dt = .1  # refresh rate
    # 机械臂两个杆的长度
    arm1l = 100
    arm2l = 100
    # 用于渲染或可视化机械臂行为的对象
    viewer = None
    # 在可视化界面中机械臂或观察点的初始位置
    viewer_xy = (400, 400)
    # 于指示是否应该通过某种方式（如鼠标点击）获取一个点
    get_point = False
    # 鼠标输入
    mouse_in = np.array([False])
    # 距离机械臂末端执行器的最小距离阈值
    point_l = 15
    # 跟踪机械臂成功抓取到某点的次数
    grab_counter = 0

    # 初始化
    def __init__(self, mode='easy'):
        # node1 (l, d_rad, x, y),
        # node2 (l, d_rad, x, y)
        # 给实例赋值
        self.mode = mode
        # 用于存储两个机械臂的信息
        self.arm_info = np.zeros((2, 4))
        # 将长度赋值给机械臂属性
        self.arm_info[0, 0] = self.arm1l
        self.arm_info[1, 0] = self.arm2l
        # 机械臂的起始位置和终点位置
        self.point_info = np.array([250, 303])
        self.point_info_init = self.point_info.copy()
        # 质心
        self.center_coord = np.array(self.viewer_xy)/2

    # 执行一步操作方法
    def step(self, action):
        # action = (node1 angular v, node2 angular v)
        # 限制action的值在角度最小和最大值之间
        action = np.clip(action, *self.action_bound)
        # 根据角速度和步长更新机械臂的角位移
        self.arm_info[:, 1] += action * self.dt
        # 限制角位移在[0,2 * pi]
        self.arm_info[:, 1] %= np.pi * 2

        # 分别提取两个机械臂的角位移
        arm1rad = self.arm_info[0, 1]
        arm2rad = self.arm_info[1, 1]
        # 分别计算两个机械臂在其当前角位移下的末端执行器的位置偏移量
        arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)])
        arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2rad), self.arm_info[1, 0] * np.sin(arm2rad)])
        # 更新每个杆件的末端坐标：
        # 第一个 基座坐标 + 偏移量
        # 第二个 基于第一个末端的坐标 + 自身偏移量
        self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy  # (x1, y1)
        self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2dx_dy  # (x2, y2)

        # 返回当前环境的状态和第二个杆件末端到某点的距离
        s, arm2_distance = self._get_state()
        # 根据第二个机械臂末端执行器的距离来计算奖励
        r = self._r_func(arm2_distance)

        # 代码返回当前的环境状态 s、奖励 r，以及 self.get_point
        return s, r, self.get_point

    # 环境重置函数
    def reset(self):
        # 重置基本状态
        # 重置后未抓取到目标点，抓取计数器被清零
        self.get_point = False
        self.grab_counter = 0

        # 根据模式设置不同初始条件
        # hard模式，随机在指定范围内生成新目标点
        if self.mode == 'hard':
            pxy = np.clip(np.random.rand(2) * self.viewer_xy[0], 100, 300)
            self.point_info[:] = pxy
        else:
            # 随机生成机器人两个臂的旋转角度
            arm1rad, arm2rad = np.random.rand(2) * np.pi * 2
            # 根据这两个角度和臂的初始长度，计算两个臂的末端位置
            self.arm_info[0, 1] = arm1rad
            self.arm_info[1, 1] = arm2rad
            arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)])
            arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2rad), self.arm_info[1, 0] * np.sin(arm2rad)])
            # 将计算出来的末端位置赋值给相应位置
            self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy  # (x1, y1)
            self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2dx_dy  # (x2, y2)
            # 将self.point_info重置为初始值self.point_info_init
            self.point_info[:] = self.point_info_init
        return self._get_state()[0]

    # 在图形界面上渲染和更新视图
    def render(self):
        # 检查是否初始化
        if self.viewer is None:
            # 初始化viewer
            self.viewer = Viewer(*self.viewer_xy, self.arm_info, self.point_info, self.point_l, self.mouse_in)
        # 渲染视图
        self.viewer.render()

    # 动作随机生成
    def sample_action(self):
        # 返回一个在指定边界内均匀分布的随机动作
        return np.random.uniform(*self.action_bound, size=self.action_dim)

    # 设置帧率（FPS，即每秒帧数）的限制
    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    # 获取当前状态的数组
    def _get_state(self):
        # return the distance (dx, dy) between arm finger point with blue point
        # 手臂末端与蓝点的距离
        arm_end = self.arm_info[:, 2:4]
        # 计算手臂末端和蓝点之间的距离
        t_arms = np.ravel(arm_end - self.point_info)
        # 中心坐标与蓝点的相对距离
        center_dis = (self.center_coord - self.point_info)/200
        # 抓取状态
        in_point = 1 if self.grab_counter > 0 else 0
        # 返回语句
        return np.hstack([in_point, t_arms/200, center_dis,
                          # arm1_distance_p, arm1_distance_b,
                          ]), t_arms[-2:]

    # 基于距离的奖励函数
    def _r_func(self, distance):
        # 阈值
        t = 50
        # 计算绝对距离
        abs_distance = np.sqrt(np.sum(np.square(distance)))
        # 奖励计算
        r = -abs_distance/200
        # 抓取逻辑
        if abs_distance < self.point_l and (not self.get_point):
            r += 1.
            self.grab_counter += 1
            if self.grab_counter > t:
                # 增加抓取成功的奖励
                r += 10.
                self.get_point = True
        elif abs_distance > self.point_l:
            # 重置计数器
            self.grab_counter = 0
            # 重置点
            self.get_point = False
        return r

# 设置一个视窗
class Viewer(pyglet.window.Window):
    # 存储背景颜色
    color = {
        'background': [1]*3 + [1]
    }
#    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    # 初始化方法
    # 使用 pyglet 库进行图形编程时显示一些图形元素（如手臂和点）而设计的
    # width 和 height 用于设置窗口大小
    # arm_info、point_info 和 point_l 可能包含与手臂和点相关的额外信息
    # mouse_in 用于处理鼠标输入的回调函数
    def __init__(self, width, height, arm_info, point_info, point_l, mouse_in):
        # 设置了窗口的宽度和高度，将窗口设置为不可调整大小（resizable=False），设置了窗口标题为 "Arm"
        super(Viewer, self).__init__(width, height, resizable=False, caption='Arm', vsync=False)  # vsync=False to not use the monitor FPS
        # 设置了窗口在屏幕上的位置，使其左上角距离屏幕左侧 80 像素，距离屏幕顶部 10 像素
        self.set_location(x=80, y=10)
        # 设置了 OpenGL 的清除颜色
        pyglet.gl.glClearColor(*self.color['background'])

        # 将传入的参数赋值给实例变量
        self.arm_info = arm_info
        self.point_info = point_info
        self.mouse_in = mouse_in
        self.point_l = point_l

        # 试图计算窗口中心点的坐标
        self.center_coord = np.array((min(width, height)/2, ) * 2)
        # 创建了一个图形批次（batch），用于高效地管理和渲染多个图形元素
        self.batch = pyglet.graphics.Batch()

        # 手臂和点的顶点坐标创建了三个空数组
        arm1_box, arm2_box, point_box = [0]*8, [0]*8, [0]*8
        # 定义了三种颜色的 RGBA 值，但每个颜色值都被重复了四次
        c1, c2, c3 = (249, 86, 86)*4, (86, 109, 249)*4, (249, 39, 65)*4
        # 添加了三个图形元素：一个点和两条手臂
        self.point = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', point_box), ('c3B', c2))
        self.arm1 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm1_box), ('c3B', c1))
        self.arm2 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm2_box), ('c3B', c1))

    # 负责处理窗口的渲染循环中的关键步骤
    def render(self):
        # 更新 pyglet 的内部时钟
        pyglet.clock.tick()
        # 更新手臂数据相关的数据和状态
        self._update_arm()
        self.switch_to()
        # 在 pyglet 中，事件可以包括键盘按键、鼠标移动、点击等
        self.dispatch_events()
        # 调用与 'on_draw' 事件相关联的事件处理函数
        self.dispatch_event('on_draw')
        # flip 方法用于交换前缓冲区和后缓冲区
        self.flip()

    # 定义窗口绘制的活动
    def on_draw(self):
        self.clear()
        self.batch.draw()
        # self.fps_display.draw()

    # 机械臂更新
    def _update_arm(self):
        # 获取手臂周围正方形的边长的一半
        point_l = self.point_l
        # 计算用于定义一个图形对象，围绕中心点的正方形顶点坐标
        point_box = (self.point_info[0] - point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] + point_l,
                     self.point_info[0] - point_l, self.point_info[1] + point_l)
        self.point.vertices = point_box

        # 计算第一个手臂的起点和终点坐标
        arm1_coord = (*self.center_coord, *(self.arm_info[0, 2:4]))  # (x0, y0, x1, y1)
        # 计算第二个手臂的起点和终点坐标，直接使用arm_info中的值
        arm2_coord = (*(self.arm_info[0, 2:4]), *(self.arm_info[1, 2:4]))  # (x1, y1, x2, y2)
        # 计算第一个手臂的“厚度”对应的旋转角度
        arm1_thick_rad = np.pi / 2 - self.arm_info[0, 1]
        # 这些点用于定义一个更宽的手臂形状 
        x01, y01 = arm1_coord[0] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] + np.sin(
            arm1_thick_rad) * self.bar_thc
        x02, y02 = arm1_coord[0] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] - np.sin(
            arm1_thick_rad) * self.bar_thc
        x11, y11 = arm1_coord[2] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] - np.sin(
            arm1_thick_rad) * self.bar_thc
        x12, y12 = arm1_coord[2] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] + np.sin(
            arm1_thick_rad) * self.bar_thc
        # 表示手臂的轮廓
        arm1_box = (x01, y01, x02, y02, x11, y11, x12, y12)
        # 对第二个手臂重复该过程
        arm2_thick_rad = np.pi / 2 - self.arm_info[1, 1]
        x11_, y11_ = arm2_coord[0] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] - np.sin(
            arm2_thick_rad) * self.bar_thc
        x12_, y12_ = arm2_coord[0] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] + np.sin(
            arm2_thick_rad) * self.bar_thc
        x21, y21 = arm2_coord[2] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] + np.sin(
            arm2_thick_rad) * self.bar_thc
        x22, y22 = arm2_coord[2] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] - np.sin(
            arm2_thick_rad) * self.bar_thc
        arm2_box = (x11_, y11_, x12_, y12_, x21, y21, x22, y22)
        # 最后，将计算出的手臂轮廓顶点坐标赋给相应的图形对象，以便在绘制时使用
        self.arm1.vertices = arm1_box
        self.arm2.vertices = arm2_box

    # 当键盘上的某个键被按下时会被调用
    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.UP:
            # 更新所有的位置值
            self.arm_info[0, 1] += .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.DOWN:
            self.arm_info[0, 1] -= .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.LEFT:
            self.arm_info[1, 1] += .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.RIGHT:
            self.arm_info[1, 1] -= .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.Q:
            pyglet.clock.set_fps_limit(1000)
        elif symbol == pyglet.window.key.A:
            pyglet.clock.set_fps_limit(30)

    def on_mouse_motion(self, x, y, dx, dy):
        self.point_info[:] = [x, y]

    def on_mouse_enter(self, x, y):
        self.mouse_in[0] = True

    def on_mouse_leave(self, x, y):
        self.mouse_in[0] = False



