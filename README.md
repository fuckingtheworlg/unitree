# Unitree Go2 EDU 道路识别赛 - 巡线工程

基于 [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python) 实现的省赛巡黄线 + 倾倒区卸料 + 岔路选径 + 上下台阶 + 充电区停车的完整方案。

> 硬件：Unitree Go2 EDU（自带 Jetson Orin Nano + 前置广角鱼眼相机）
> 开发机：MacBook Air M4 (macOS arm64)
> 运行机：狗内置 Orin Nano (Ubuntu 20.04 aarch64)
> 通信：CycloneDDS over Ethernet

## 目录结构

```
unitree/
├─ config/params.yaml           # 全局可调参数 (HSV/PID/相机/路径)
├─ src/
│  ├─ main.py                   # 入口
│  ├─ robot/                    # 与 Go2 SDK 的薄封装
│  ├─ vision/                   # 黄线 + 地标识别
│  ├─ control/                  # PID + 状态机
│  └─ utils/                    # 日志 + 离线回灌
├─ scripts/
│  ├─ record_camera.py          # 录制狗前置相机视频
│  └─ tune_hsv.py               # 实时调 HSV 阈值
├─ tests/                       # 离线单元测试
└─ deploy.sh                    # rsync 部署到狗
```

## 快速开始

### 1. Mac 端环境（开发用）

```bash
bash setup_mac.sh
```

会做：编译 cyclonedds 0.10.x → 装 venv → 装 Python 依赖 → `pip install -e` 安装 unitree_sdk2_python。
**所有依赖都装在项目本地 `.venv` 和 `.deps`**，不污染系统 Python。

之后每次开发前：

```bash
source .venv/bin/activate
export CYCLONEDDS_HOME="$(pwd)/.deps/cyclonedds/install"
```

> 启动时如果报 `Could not locate cyclonedds` 或 `ModuleNotFoundError: cyclonedds`，
> 100% 是上面的 `CYCLONEDDS_HOME` 没 export。可以加进 `~/.zshrc` 别名一键进环境。

### 2. Go2 端环境（运行用）

将本仓库 rsync 到狗的 Orin Nano：

```bash
bash deploy.sh
```

然后 SSH 进狗执行：

```bash
ssh unitree@192.168.123.18
cd ~/unitree
bash setup_robot.sh   # 仅首次需要
```

### 3. 网络配置

把 Mac/狗用网线直连（或经交换机），狗的固定 IP 是 `192.168.123.18`。
Mac 网卡 IP 设为 `192.168.123.222/24`，确认 `ping 192.168.123.18` 通。

确认狗的网卡名（一般是 `eth0`，从 Orin 上 `ip addr` 查）：
config/params.yaml 里 `network.interface` 填这个名字。

### 4. 离线调试（在 Mac 上）

```bash
# 滑动条调 HSV 阈值
python -m scripts.tune_hsv tests/data/sample.mp4

# 跑离线视频回灌测试视觉管线
python -m src.utils.video_replay tests/data/sample.mp4
```

### 5. 上狗实战

> 让狗站在起点，朝向第一段黄道，机身正前对准道路。

```bash
ssh unitree@192.168.123.18
cd ~/unitree
python -m src.main --network eth0 --province  # 省赛模式
```

按 `Ctrl+C` 紧急停止，会自动让狗 BalanceStand。

## 状态机概览

```
INIT → STAND_UP → FOLLOW_LANE
       ↓
   DETECT_DUMP_ZONE → APPROACH_DUMP → DUMP_ACTION → RESUME_FOLLOW
                                                      ↓
                                  DETECT_FORK → CHOOSE_SHORTEST → FOLLOW_LANE
                                                                   ↓
                                              DETECT_STAIR → CLIMB_STAIR
                                                              ↓
                                                          APPROACH_DOCK → DOCK → DONE
```

## 调参清单

省赛前必须现场调过的参数（写在 `config/params.yaml`）：

- `vision.yellow_hsv_lower / upper`：黄色阈值（不同光线差别很大）
- `control.lateral_pid.{kp, ki, kd}`：横向 PID
- `control.forward_speed.{follow, dump_approach, stair, dock}`：各阶段前进速度
- `landmark.dump_min_radius_ratio`：倾倒区圆环判定阈值
- `landmark.fork_branch_angle_deg`：岔路角度判定
- `landmark.stair_distance_trigger_m`：检测到台阶接近后切到爬楼步态的距离

详见 `config/params.yaml` 的注释。

## 重要安全说明

1. 所有 `Move(vx, vy, vyaw)` 调用都通过 `Go2Client.safe_move()` 走，超过阈值会被夹断。
2. 程序启动会先 `BalanceStand`，结束/异常会先 `StopMove` 再退出。
3. **首次跑务必在身上系绳/狗周围 1m 内有人随时拍急停**。
