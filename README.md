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
  --camera-width 640 --camera-height 480 --camera-fps 30 \
  --tactile-source xense \
  --xense-left-serial OG001783 --xense-right-serial OG001782
```

默认机器人端口 6001，腕部/基座相机端口 5000/5001，左右触觉端口 5100/5101。
仅做不依赖相机的推理时，服务端可加 `--no-enable-camera`。
FTS-300 由采集/推理客户端按原有参数直接读取，不由启动器另建节点。

采集仍使用原脚本入口；默认新数据写入本副本的 `data/ur5_tactile_30hz`，
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

## 通讯与硬件优化

移植自 [Asgod-1/ur5-rg2-tactile-force 的 273a317 提交](https://github.com/Asgod-1/ur5-rg2-tactile-force/commit/273a3172ee616511c8df0f77f9c221036db409c3)，
按本项目的 `gello/`、`real_ur5rg2/` 目录适配。螺母/螺栓标定、随机复位、力与触觉限值、
`distributed_force` 字段、指尖中心相对螺母位姿字段及 `L` 自动采集流程保持兼容。

- RG2 脚本由后台线程执行，只保留最新待执行目标；夹爪忙时暂停手臂命令，失败时报告错误。
  自动采集等待脚本完成再推进闭合/打开阶段。ZMQ 客户端默认仍等待夹爪完成，兼容原有两份推理；
  采集器显式使用 `blocking_gripper=False`，开合期间可继续录制。
- FT300-S 新增默认 `100 Hz` 连续流，串口为 `19200` 波特率，逐包验证 CRC、错包重同步。
  打开后立即后台接收，避免等待续采选择或去零期间积压。可选 FTDI 低延迟模式在关闭时恢复；
  `--force-source modbus-serial` 可使用原问答模式。后台样本默认超过 `0.1 s` 或读取失败时，
  跳过该轮运动和记录，不使用旧力数据。
- 双 Xense 默认 `--xense-backend process`，每个传感器在独立的 `spawn` 子进程中创建、读取、释放 SDK，
  避免两个传感器共用 SDK 锁造成串行推理和帧超时。共享内存只保留最新结果，不排队积压旧图；
  SDK 异常或子进程退出会向采集端报错。子进程内限制 OpenCV 线程为 `1`，
  ONNX 会话使用 intra-op `6`、inter-op `1` 并关闭自旋。
  可用 `--xense-backend thread` 回到原公平排队线程模式作对照。
  SDK 支持 `CaptureTimeStamp` 时以该时间去重；本机 `xensesdk==2.1.3` 不提供此字段，
  兼容模式使用 `TimeStamp`，并对 `Rectify` 图像摘要去重：相同图像不会因重复推理而刷新有效期。
  这是软件帧时间估计，不是硬件采样时间，不能据此承诺严格同步；完全相同的新图也会保守地视为重复帧。
  缺少有效图像/时间戳时报告错误，不使用当前主机时间伪装为新帧。
  录制最多回看上一帧，控制与保护使用最新触觉。
  停更或捕获结果超过 `0.15 s` 时报告错误，可通过启动器的 `--xense-max-age-s` 调整。
- 相机默认 `640×480 @ 30 fps`；采集使用 RGB 专用接口并携带原生帧时间，省去零深度占位图的分配和传输。
  兼容原 RGB/深度接口，以及 `--camera-backend gello` / `realsense`。
- 录制默认以触觉捕获时间（禁用触觉时以 RGB 时间）配对各路短历史，每路距参考时间最多 `20 ms`。
  UR 状态用原生控制器时间去重和映射，配套记录同一状态包内的 TCP 位姿，避免指尖标签错配。
  缺少新帧时跳行，允许有限次数重选，不重复旧帧补足标称频率。
- 新采与续采都使用后台无损 PNG 写图，默认压缩级别 `1`；图像像素不变，文件通常更大。
  `--image-png-compress-level 6` 可恢复较高压缩。ZMQ 超时后重建请求连接，当前调用报错，
  不自动重发可能已经执行的运动命令；退出时释放后台线程、串口、相机、RTDE 和 ZMQ 资源。

更新后重启 `launch_nodes.py` 和采集进程。新采示例：

```bash
python real_ur5rg2/collect_lerobot_xenstactile_servo_control.py \
  --root ./data/ur5_tactile_30hz --fps 30 \
  --image-width 640 --image-height 480 \
  --force-serial-port /dev/ttyUSB0 --no-preview
```

按 `P` 后先等待 `Recording ready` 再开始任务动作；`L` 自动流程在闭合等待结束后保存首条有效配对数据。
保存 episode 时附加 `meta/tactile_timing/`、`meta/sensor_timing/` 和 `meta/recording_performance/`，
用于检查真实帧率、时差、预热、跳行和阶段耗时，不改变训练使用的标准 `timestamp`。
软件时间配对包含主机时钟和传输延迟估计，不等同于物理采样同步，实际帧率需在本机实测。

续采已有数据时显式传入原 `--root`、帧率及图像尺寸；原默认配置可用 `--fps 20 --image-size 256`。
尺寸或帧率不匹配时会报错，不修改旧数据。可用 `--no-recording-time-alignment` 关闭时间配对，
`--no-tactile-recording-buffer` 关闭触觉录制缓冲，便于按原行为对照。

无实机回归测试（需要现有项目环境中的 pytest）：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_hardware_sync.py
```


## 其他采集与推理入口

`real_ur5rg2/collect_lerobot_smolvla_real_xenstactile_w_force.py` 同样接入后台力采样、
异步夹爪、RGB 专用传输、触觉捕获缓冲、软件时间配对、续采后台 PNG 和性能记录。
保留该入口原有的 `.force` 数据字段、标定和手动 `L` 复位行为；原伺服采集入口继续使用
`.distributed_force` 和原有的自动 `L` 流程，两种数据字段不会相互改名。
新入口默认以 `30 Hz`、`640×480` 保存到 `./data/ur5_smolvla_30hz`。
续采旧数据时指定原目录及原参数，例如 `--fps 20 --image-size 256 --resume`。

两份 `tactile_policy/infer_tactile_servo_nutbolt_miben_lhy*.py` 也支持默认
`--force-source fts300-stream`，以及 `--force-source modbus-serial` 的后台采样。
它们沿用原有 `--use-force-safety` 开关：启用时才打开力传感器，关闭时不打开串口。
采样从力读取器初始化时启动，校准、等待执行确认和推理期间持续接收；主循环读取最新缓存，
默认超过 `--force-max-age-s 0.1` 或后台报错时停止正常推理命令。
`--force-serial-low-latency` 可控制连续流的适配器低延迟模式，退出时恢复并关闭串口。
Modbus TCP / UR RTDE 力源、原有控制频率、阈值、标定和动作阶段保持原实现。
双触觉在转换图像前先读取两路快照；夹爪仍使用同步等待，以兼容原推理阶段。

同一力串口由当前采集或推理进程占用；切换入口前先退出占用该串口的进程。
补充入口回归测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_entrypoint_sync.py
```
