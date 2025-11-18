import time
import sys
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

CONTROL_DT = 0.01  # 100 Hz
NUM_MOTORS = 29     # G1 电机总数

# 默认关节角度（rad），按 g1_29dof_anneal_23dof.yaml 的顺序（23 DOF）
DEFAULT_POS = np.array([
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # 左腿 0-5
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,   # 右腿 6-11
    0.0, 0.0, 0.0,                    # 腰部 12-14
    0.0, 0.0, 0.0, 0.0,               # 左臂 15-18
    0.0, 0.0, 0.0, 0.0                # 右臂 19-22
])

# PD 参数（与训练一致）
KP = np.array([
    100, 100, 100, 200, 20, 20,
    100, 100, 100, 200, 20, 20,
    400, 400, 400,
    90, 60, 20, 60,
    90, 60, 20, 60
])

KD = np.array([
    2.5, 2.5, 2.5, 5.0, 0.2, 0.1,
    2.5, 2.5, 2.5, 5.0, 0.2, 0.1,
    5.0, 5.0, 5.0,
    2.0, 1.0, 0.4, 1.0,
    2.0, 1.0, 0.4, 1.0
])

# 映射：23 个关节索引 -> 29 个电机索引
MOTOR_MAP = list(range(12)) + list(range(12, 15)) + list(range(15, 19)) + list(range(22, 26))
ACTION_IDX = list(range(23))

class PDStandController:
    def __init__(self, net_iface=None):
        self.net_iface = net_iface
        self.low_cmd = LowCmd_()
        self.low_state = None
        self.crc = CRC()
        self.mode_machine = 0

    def init(self):
        if self.net_iface:
            ChannelFactoryInitialize(0, self.net_iface)
        else:
            ChannelFactoryInitialize(0)

        # Motion switcher
        msc = MotionSwitcherClient()
        msc.SetTimeout(5.0)
        msc.Init()

        status, result = msc.CheckMode()
        while result.get("name"):
            print(f"检测到 {result['name']}，正在释放...")
            msc.ReleaseMode()
            time.sleep(1)
            status, result = msc.CheckMode()
        print("已切换到低级控制模式")

        self.mode_machine = result.get("mode", 0)

        # Publisher/subscriber
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()

        self.sub = ChannelSubscriber("rt/lowstate", LowState_)
        self.sub.Init(self.state_callback, 10)

        # 初始化命令
        self.low_cmd.mode_pr = 0
        self.low_cmd.mode_machine = self.mode_machine
        for motor in self.low_cmd.motor_cmd:
            motor.mode = 1
            motor.q = 0.0
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = 0.0
            motor.kd = 0.0
            motor.reserve = 0
        self.low_cmd.reserve = [0, 0, 0, 0]
        self.low_cmd.crc = 0

    def state_callback(self, msg: LowState_):
        self.low_state = msg

    def run(self):
        print("开始 PD 站立控制，按 Ctrl+C 退出")
        try:
            while True:
                if self.low_state is None:
                    time.sleep(0.005)
                    continue

                for motor_idx, action_idx in zip(MOTOR_MAP, ACTION_IDX):
                    target = DEFAULT_POS[action_idx]
                    motor = self.low_cmd.motor_cmd[motor_idx]
                    motor.q = target
                    motor.dq = 0.0
                    motor.tau = 0.0
                    motor.kp = KP[action_idx]
                    motor.kd = KD[action_idx]

                self.low_cmd.crc = self.crc.Crc(self.low_cmd)
                self.pub.Write(self.low_cmd)

                time.sleep(CONTROL_DT)

        except KeyboardInterrupt:
            print("\n停止 PD 控制，机器人保持当前状态")
        finally:
            pass  # 可以在这里添加 pub.Close() 等

if __name__ == "__main__":
    iface = sys.argv[1] if len(sys.argv) > 1 else None
    controller = PDStandController(iface)
    controller.init()
    controller.run()