#!/usr/bin/env python

"""
Android Phone Teleoperation 仿真测试脚本
 
功能：
1. 创建仿真机器人
2. 启动 WebXR 服务器
3. 从 Android 手机接收控制信号
4. 实时显示数据流
 
依赖：
- teleop: WebXR teleoperation 库
- lerobot: 机器人控制框架
- tests.mocks.mock_robot: 仿真机器人
"""

import time
import numpy as np
from teleop import Teleop
from pathlib import Path
 
# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
 
from tests.mocks.mock_robot import MockRobot, MockRobotConfig
 
FPS = 30
 
def main():
    print("=== Android Phone Teleoperation 仿真测试 ===\n")
    
    # 1. 创建仿真机器人
    robot_config = MockRobotConfig(
        n_motors=6,
        id="sim_piper",
        random_values=False,
        static_values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    robot = MockRobot(robot_config)
    robot.connect()
    print(f"✓ 仿真机器人已连接: {robot.name}")
    print(f"  - 关节数量: {len(robot.motors)}")
    print(f"  - Action features: {robot.action_features}")
    print(f"  - Observation features: {robot.observation_features}")
    print()
    
    # 2. 创建 Phone teleoperator（Android）
    from lerobot.teleoperators.phone.config_phone import PhoneConfig, PhoneOS
    from lerobot.teleoperators.phone.teleop_phone import Phone
    
    teleop_config = PhoneConfig(phone_os=PhoneOS.ANDROID)
    teleop = Phone(teleop_config)
    
    print("="*50)
    print("正在启动 WebXR 服务器...")
    print("="*50 + "\n")
    
    # 3. 连接 Phone（会自动启动 WebXR 服务器并打印 URL）
    teleop.connect()
    
    print("\n" + "="*50)
    print("✓ WebXR 服务器已启动！")
    print("在手机浏览器中打开上面显示的 URL")
    print("按住 'Move' 按钮开始控制")
    print("="*50 + "\n")
    
    # 4. 测试循环（纯控制，无可视化）
    try:
        frame_count = 0
        last_print_time = time.time()
        print("等待 Phone 连接...")
        
        while True:
            t0 = time.perf_counter()
            
            # 获取机器人观测
            robot_obs = robot.get_observation()
            
            # 获取 Phone 动作
            phone_action = teleop.get_action()
            
            # 检查是否启用
            enabled = phone_action.get("phone.enabled", False)
            pos = phone_action.get("phone.pos")
            
            # 如果启用，发送动作到机器人
            if enabled and pos is not None:
                # 简单映射（真实场景会用 IK）
                robot_action = {
                    "motor_1.pos": pos[0] * 100,
                    "motor_2.pos": pos[1] * 100,
                    "motor_3.pos": pos[2] * 100,
                    "motor_4.pos": 0.0,
                    "motor_5.pos": 0.0,
                    "motor_6.pos": 0.0,
                }
                robot.send_action(robot_action)
            
            # 每 3 秒打印一次状态
            current_time = time.time()
            if current_time - last_print_time >= 3.0:
                frame_count += 1
                elapsed_frame = time.perf_counter() - t0
                fps = 1.0 / elapsed_frame if elapsed_frame > 0 else 0
                
                print(f"Frame {frame_count}:")
                print(f"  - enabled: {enabled}")
                print(f"  - fps: {fps:.1f}")
                
                if pos is not None and enabled:
                    print(f"  - Phone pos: [{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]")
                    print(f"  - Robot action:")
                    for motor, val in robot_action.items():
                        print(f"    {motor}: {val:8.2f}")
                else:
                    print(f"  - Phone pos: None (disabled)")
                
                last_print_time = current_time
            
            # 保持帧率
            elapsed = time.perf_counter() - t0
            sleep_time = max(1.0 / FPS - elapsed, 0.0)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n\n用户中断，退出...")
    
    finally:
        print("\n" + "="*60)
        print("正在断开连接...")
        robot.disconnect()
        teleop.disconnect()
        print("✓ 设备已断开连接")
        print("="*60)
 
 
if __name__ == "__main__":
    main()