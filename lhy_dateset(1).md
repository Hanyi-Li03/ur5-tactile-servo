## 第一次采集
episode: 0-931
RESET_TCP_POSITION_RANDOM_RANGE_X = 0.002

## 第二次采集
episode: 0-931
RESET_TCP_POSITION_RANDOM_RANGE_X = 0.003

## 第三次采集
episode:real_ur5rg2/data/ur5_rg2_real_smolvla_dataset_xense_tactile_boltnut_demo_L1/data/chunk-004/episode_004833.parquet
RESET_TCP_POSITION_RANDOM_RANGE_X = 0
base_rotation @ tcp_y_rotation
RESET_TCP_Y_ROTATION_RANGE_DEG = 2


## 第四次采集
real_ur5rg2/data/ur5_rg2_real_smolvla_dataset_xense_tactile_boltnut_demo_L1/data/chunk-004/episode_004846.parquet
RESET_TCP_POSITION_RANDOM_RANGE_Y = 0.002

RESET_TCP_POSITION_RANDOM_RANGE_Z = 0.003
RESET_TCP_Y_ROTATION_MIN_DEG = 0.2
RESET_TCP_Y_ROTATION_RANGE_DEG = 0.5

## 第五次采集 2026-09-19
### L2 : 0 - 1254 

RESET_TCP_POSITION_RANDOM_RANGE_X = 0.000
RESET_TCP_POSITION_RANDOM_RANGE_Y = 0.000
RESET_TCP_POSITION_RANDOM_RANGE_Z = 0.000
RESET_TCP_Y_ROTATION_MIN_DEG = 0.2
RESET_TCP_Y_ROTATION_RANGE_DEG = 0.5
RESET_TCP_X_ROTATION_RANGE_DEG = 8.0


## 第六次采集 2026-09-20
### L2 : 1254 -  2253

RESET_TCP_POSITION_RANDOM_RANGE_X = 0.000
RESET_TCP_POSITION_RANDOM_RANGE_Y = 0.000
RESET_TCP_POSITION_RANDOM_RANGE_Z = 0.002
RESET_TCP_Y_ROTATION_MIN_DEG = 0.2
RESET_TCP_Y_ROTATION_RANGE_DEG = 0.4
RESET_TCP_X_ROTATION_RANGE_DEG = 10.0

## 第七次采集 2026-09-20
### L2 : 2253 -  

RESET_TCP_POSITION_RANDOM_RANGE_X = 0.000
RESET_TCP_POSITION_RANDOM_RANGE_Y = 0.000
RESET_TCP_POSITION_RANDOM_RANGE_Z = 0.003
RESET_TCP_Y_ROTATION_MIN_DEG = 0.2
RESET_TCP_Y_ROTATION_RANGE_DEG = 0.4
RESET_TCP_X_ROTATION_RANGE_DEG = 12.0

## 第七次采集 2026-09-21
### L3 : 0-
RESET_TCP_POSITION_RANDOM_RANGE_X = 0.000
RESET_TCP_POSITION_RANDOM_RANGE_Y = 0.002
RESET_TCP_POSITION_RANDOM_RANGE_Z = 0.003
RESET_TCP_Y_ROTATION_MIN_DEG = 0.2
RESET_TCP_Y_ROTATION_RANGE_DEG = 0.4
RESET_TCP_X_ROTATION_RANGE_DEG = 15.0

## 第八次采集 2026-10-9
real_ur5rg2/data/ur5_rg2_real_smolvla_dataset_xense_tactile_boltnut_demo_L5000
RESET_TCP_POSITION_RANDOM_RANGE_X = 0.000
RESET_TCP_POSITION_RANDOM_RANGE_Y = 0.001
RESET_TCP_POSITION_RANDOM_RANGE_Z = 0.005
RESET_TCP_Y_ROTATION_MIN_DEG = 0.2
RESET_TCP_Y_ROTATION_RANGE_DEG = 0.4
RESET_TCP_X_ROTATION_RANGE_DEG = 5.0

## 第九次采集 2026-10-9
real_ur5rg2/data/ur5_rg2_real_smolvla_dataset_xense_tactile_boltnut_demo_L5000v2
RESET_TCP_POSITION_RANDOM_RANGE_X = 0.000
RESET_TCP_POSITION_RANDOM_RANGE_Y = 0.002
RESET_TCP_POSITION_RANDOM_RANGE_Z = 0.005
RESET_TCP_Y_ROTATION_MIN_DEG = 0.2
RESET_TCP_Y_ROTATION_RANGE_DEG = 0.0
RESET_TCP_X_ROTATION_RANGE_DEG = 7.0

## 第十次采集 2026-10-15
real_ur5rg2/data/ur5_rg2_real_smolvla_dataset_xense_tactile_boltnut_demo_L7 2000
RESET_TCP_POSITION_RANDOM_RANGE_X = 0.000
RESET_TCP_POSITION_RANDOM_RANGE_Y = 0.002
RESET_TCP_POSITION_RANDOM_RANGE_Z = 0.004
RESET_TCP_Y_ROTATION_MIN_DEG = 0.2
RESET_TCP_Y_ROTATION_RANGE_DEG = 0.0
RESET_TCP_X_ROTATION_MIN_DEG = 7.0
RESET_TCP_X_ROTATION_RANGE_DEG = 15.0

## 第十次采集 2026-10-15
real_ur5rg2/data/ur5_rg2_real_smolvla_dataset_xense_tactile_boltnut_demo_L8 2000
RESET_TCP_POSITION_RANDOM_RANGE_X = 0.000
RESET_TCP_POSITION_RANDOM_RANGE_Y = 0.002
RESET_TCP_POSITION_RANDOM_RANGE_Z = 0.002
RESET_TCP_Y_ROTATION_MIN_DEG = 0.1
RESET_TCP_Y_ROTATION_RANGE_DEG = 0.0
RESET_TCP_X_ROTATION_MIN_DEG = 15.0
RESET_TCP_X_ROTATION_RANGE_DEG = 20.0




python launch_nodes.py   --robot ur   --camera-width 424   --camera-height 240   --camera-fps 15   --tactile-source xense   --xense-left-serial OG001783   --xense-right-serial OG001782

sudo chmod 777 '/dev/ttyUSB0'

python real_ur5rg2/collect_lerobot_xenstactile_servo_control.py   --task "Insert the nut into the bolt."   --num-episodes 20   --no-preview   --no-force-safety   --collect-tactile   --force-serial-port /dev/ttyUSB0   --force-serial-timeout-s 0.1   --force-serial-retries 5


python real_ur5rg2/collect_lerobot_smolvla_real_xenstactile_w_force_L.py --task "Insert the nut into the bolt."   --num-episodes 1000   --no-preview   --no-force-safety   --collect-tactile   --force-serial-port /dev/ttyUSB0   --force-serial-timeout-s 0.1   --force-serial-retries 5

python real_ur5rg2/infer_tactile_servo_nutbolt_miben_lhy.py  --execute  --mode full-cycle
python real_ur5rg2/infer_tactile_servo_nutbolt_miben_lhy.py  --execute  --grasp-placement-test  --mode full-cycle
python real_ur5rg2/infer_tactile_servo_nutbolt_miben_lhy.py  --execute  --mode validate-pose --validation-rounds 1




RESET_TCP_POSITION_RANDOM_RANGE_X = 0.000
RESET_TCP_POSITION_RANDOM_RANGE_Y = 0.002
RESET_TCP_POSITION_RANDOM_RANGE_Z = 0.004
RESET_TCP_X_ROTATION_RANGE_DEG = 5.0
RESET_TCP_Y_ROTATION_MIN_DEG = 0.2
RESET_TCP_Y_ROTATION_RANGE_DEG = 0.0