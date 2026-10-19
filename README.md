# UR5 tactile servo

UR5 + RG2 + 双 Xense 的实机采集与螺母/螺栓触觉伺服精简副本。
本仓库不包含数据集、checkpoint、训练/离线评估脚本、仿真、ROS 或模型资产。
`.gitignore` 排除各级 `data/`、数据集文件、模型权重、运行输出、缓存、虚拟环境和本机配置，
这些文件保留在本地，不纳入版本控制。

## 保留的入口

| 文件 | 用途 |
| --- | --- |
| `launch_nodes.py` | 启动 UR/RG2、RealSense、左右 Xense 的 ZMQ 服务 |
| `real_ur5rg2/collect_lerobot_xenstactile_servo_control.py` | 现有键盘采集与机械臂控制 |
| `tactile_policy/infer_tactile_servo_nutbolt_miben_lhy.py` | 原版触觉伺服推理 |
| `tactile_policy/infer_tactile_servo_nutbolt_miben_lhy_new.py` | 当前 new 版触觉伺服推理 |

`gello/` 仅保留这些入口需要的 UR/RG2、相机、Xense、FTS-300 和 ZMQ 通讯模块。
`tactile_policy/model.py`、`data.py` 提供推理所需模型结构、归一化及预处理；
`dataset_delete_utils.py` 是采集时覆盖已有输出所需的确认工具。

## 环境

优先复用本机已安装好的环境，Python 3.10：

```bash
conda activate lerobot_mujoco
cd /home/lhy/ur5-tactile-servo
```

`requirements.txt` 记录当前环境的主要依赖版本，供以后重建环境参考。
LeRobot 必须兼容旧的 `lerobot.common.*` API，不能直接假定最新版兼容。
Torch 的 CUDA 安装源需按新机器调整；这里不复制整个 conda 环境。
Xense 使用厂商完整安装的 `xensesdk==2.1.3`，以及设备对应的 SDK 配置/标定。
这些不属于项目源码，未复制；仍需正确的 USB/串口权限。

## 使用

查看参数不会连接硬件：

```bash
python launch_nodes.py --help
python real_ur5rg2/collect_lerobot_xenstactile_servo_control.py --help
python tactile_policy/infer_tactile_servo_nutbolt_miben_lhy.py --help
python tactile_policy/infer_tactile_servo_nutbolt_miben_lhy_new.py --help
```

实际启动服务的示例（会连接硬件，先确认 IP 和各设备序列号）：

```bash
python launch_nodes.py \
  --robot ur --robot-ip 192.168.1.102 \
  --camera-backend lerobot \
  --camera-width 424 --camera-height 240 --camera-fps 15 \
  --tactile-source xense \
  --xense-left-serial OG001783 --xense-right-serial OG001782
```

默认机器人端口 6001，腕部/基座相机端口 5000/5001，左右触觉端口 5100/5101。
仅做不依赖相机的推理时，服务端可加 `--no-enable-camera`。
FTS-300 由采集/推理客户端按原有参数直接读取，不由启动器另建节点。

采集仍使用原脚本及参数；默认新数据写入本副本的 `data/ur5_tactile`，
也可通过 `--root /绝对路径/新数据集` 指定其他目录。采集默认会控制机械臂，
需要事先核对标定位置、工作空间、力限制和串口。不要用 `--overwrite` 指向要保留的数据。

两份推理都必须显式提供外部权重，例如查看该权重对应的原有参数后运行：

```bash
python tactile_policy/infer_tactile_servo_nutbolt_miben_lhy_new.py \
  --checkpoint /绝对路径/best_model.pt --mode full-cycle
```

不带 `--execute` 不发送推理运动命令，但仍会连接并读取硬件；这不是纯离线测试。
真实执行需明确添加 `--execute`，并事先复核当前装夹、TCP/螺栓标定和安全条件。
原版推理用同样方式调用不带 `_new` 的文件。

## 相对原项目的整理范围

- 保留复制时三份主脚本的控制/采集逻辑、标定值和安全阈值；本次不修改力触发或开拧判据。
- 将启动器集中到根目录，仅保留当前需要的 UR/RG2、RealSense 和 Xense 分支。
- 移除两份推理里绑定旧项目的默认权重绝对路径，改为必填 `--checkpoint`。
- 将采集默认输出路径绑定到本副本，修正本地工具导入与启动器提示路径。
- 删除 UR 模块里未使用的其他夹爪导入，不复制不相关设备代码。

迁移检查只执行语法、导入、命令行帮助及模拟对象检查，不代表已完成实机安全验证。

