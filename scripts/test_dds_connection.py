"""
DDS连接测试脚本
用于验证与G1机器人的DDS通信是否正常
"""
import time
import sys
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

def test_dds_connection(network_interface=None):
    """
    测试DDS连接
    
    测试内容：
    1. 初始化DDS通道
    2. 订阅机器人状态（rt/lowstate）
    3. 检查是否能收到状态数据
    4. 测试运动切换器（MotionSwitcher）
    5. 测试发布控制命令（rt/lowcmd）
    """
    print("="*80)
    print("DDS连接测试")
    print("="*80)
    
    # 1. 初始化DDS通道
    print("\n[1/5] 初始化DDS通道...")
    try:
        if network_interface:
            ChannelFactoryInitialize(0, network_interface)
            print(f"   ✅ 使用网络接口: {network_interface}")
        else:
            ChannelFactoryInitialize(0)
            print("   ✅ 使用默认网络接口")
    except Exception as e:
        print(f"   ❌ 初始化失败: {e}")
        return False
    
    # 2. 测试订阅机器人状态
    print("\n[2/5] 测试订阅机器人状态 (rt/lowstate)...")
    state_received = False
    state_count = 0
    
    def state_handler(msg: LowState_):
        nonlocal state_received, state_count
        state_received = True
        state_count += 1
        if state_count == 1:
            print(f"   ✅ 收到第一个状态数据！")
            print(f"      - IMU RPY: [{msg.imu_state.rpy[0]:.4f}, {msg.imu_state.rpy[1]:.4f}, {msg.imu_state.rpy[2]:.4f}]")
            print(f"      - 电机数量: {len(msg.motor_state)}")
            print(f"      - 模式: {msg.mode_machine}")
    
    try:
        state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        state_sub.Init(state_handler, 10)
        print("   ✅ 状态订阅者创建成功")
        
        # 等待接收状态（最多等待5秒）
        print("   ⏳ 等待接收状态数据（最多5秒）...")
        timeout = 5.0
        start_time = time.time()
        while not state_received and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if state_received:
            print(f"   ✅ 状态接收正常！已收到 {state_count} 个状态数据包")
        else:
            print("   ⚠️  超时：未收到状态数据")
            print("      可能原因：")
            print("      1. 机器人未开机或未连接到网络")
            print("      2. 网络接口名称不正确")
            print("      3. 防火墙阻止了DDS通信")
            return False
            
    except Exception as e:
        print(f"   ❌ 状态订阅失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 测试运动切换器
    print("\n[3/5] 测试运动切换器 (MotionSwitcher)...")
    try:
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()
        print("   ✅ 运动切换器初始化成功")
        
        # 检查当前模式
        status, result = msc.CheckMode()
        if status:
            print(f"   ✅ 当前模式: {result.get('name', 'Unknown')}")
            if result.get('name'):
                print(f"   ⚠️  检测到高级控制模式，部署时需要释放")
        else:
            print("   ✅ 当前处于低级控制模式（适合部署）")
            
    except Exception as e:
        print(f"   ⚠️  运动切换器测试失败: {e}")
        print("      这可能不影响部署，但建议检查")
    
    # 4. 测试发布控制命令
    print("\n[4/5] 测试发布控制命令 (rt/lowcmd)...")
    try:
        cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        cmd_pub.Init()
        print("   ✅ 控制命令发布者创建成功")
        
        # 创建一个测试命令（不实际发送，只测试消息结构）
        test_cmd = unitree_hg_msg_dds__LowCmd_()
        test_cmd.mode_pr = 0
        test_cmd.mode_machine = 0
        for motor in test_cmd.motor_cmd:
            motor.mode = 0
            motor.q = 0.0
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = 0.0
            motor.kd = 0.0
            motor.reserve = 0
        test_cmd.reserve = [0, 0, 0, 0]
        test_cmd.crc = 0
        print("   ✅ 控制命令消息创建成功（未发送）")
        print("   ⚠️  注意：未实际发送命令，仅测试通道创建")
        
    except Exception as e:
        print(f"   ❌ 控制命令发布失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. 持续监控状态（可选）
    print("\n[5/5] 持续监控状态（10秒）...")
    print("   按 Ctrl+C 提前结束")
    try:
        start_time = time.time()
        last_count = state_count
        while time.time() - start_time < 10.0:
            time.sleep(1.0)
            current_count = state_count
            rate = current_count - last_count
            print(f"   状态接收速率: {rate} Hz (累计: {current_count} 个数据包)")
            last_count = current_count
    except KeyboardInterrupt:
        print("\n   用户中断")
    
    # 总结
    print("\n" + "="*80)
    print("测试总结")
    print("="*80)
    print(f"✅ DDS通道初始化: 成功")
    print(f"{'✅' if state_received else '❌'} 状态订阅: {'成功' if state_received else '失败'}")
    print(f"✅ 控制命令发布: 成功")
    print(f"📊 状态数据包总数: {state_count}")
    
    if state_received:
        print("\n🎉 DDS连接测试通过！可以开始部署策略了。")
        return True
    else:
        print("\n⚠️  DDS连接测试未完全通过，请检查上述问题。")
        return False

def test_helloworld():
    """
    测试基础的DDS发布/订阅（不依赖机器人）
    """
    print("="*80)
    print("基础DDS测试（HelloWorld示例）")
    print("="*80)
    print("\n这个测试不需要机器人，用于验证DDS基础功能。")
    print("请打开两个终端，分别运行：")
    print("  终端1: python3 test_dds_connection.py --helloworld-pub")
    print("  终端2: python3 test_dds_connection.py --helloworld-sub")
    print("\n如果能看到消息传递，说明DDS基础功能正常。")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        if '--help' in sys.argv or '-h' in sys.argv:
            print("使用方法:")
            print("  python3 test_dds_connection.py [网络接口名称]")
            print("\n示例:")
            print("  python3 test_dds_connection.py wlp0s20f3")
            print("\n提示: 使用 'ifconfig' 或 'ip addr' 查看网络接口名称")
            sys.exit(0)
        elif '--helloworld' in sys.argv:
            test_helloworld()
            sys.exit(0)
        else:
            network_interface = sys.argv[1]
    else:
        network_interface = None
        print("⚠️  未指定网络接口，使用默认设置")
        print("   建议使用: python3 test_dds_connection.py <网络接口名称>")
        print("   使用 'ifconfig' 查看网络接口名称\n")
    
    success = test_dds_connection(network_interface)
    sys.exit(0 if success else 1)

