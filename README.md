# Mobile Robot DRL Navigation — ROS 2 Jazzy + Gazebo Harmonic

DRL (DDPG, TD3) navigation for a TurtleBot3 Burger in a moving-obstacle arena,
ported to **ROS 2 Jazzy** and **Gazebo Harmonic (gz-sim 8)** with the official
TurtleBot3 meshes. Includes a live SLAM map, ground-truth TF, and an RViz setup
for inspection.

<p align="center">
  <img src="media/drl_tbot3.gif" width="720" alt="DRL TurtleBot3 simulation" />
</p>

## Contents
* [Requirements](#requirements)
* [Install](#install)
* [Build](#build)
* [Run the demo](#run-the-demo)
* [What's in the stack](#whats-in-the-stack)
* [Switching models / algorithms](#switching-models--algorithms)
* [Package layout](#package-layout)
* [Troubleshooting](#troubleshooting)
* [Acknowledgments](#acknowledgments)

## Requirements
* Ubuntu 24.04
* ROS 2 Jazzy (`ros-jazzy-ros-base` or desktop)
* Gazebo Harmonic (`gz-sim 8`) shipped via `ros-jazzy-gz-*` vendor packages
* NVIDIA GPU recommended (PyTorch + CUDA)

## Install
Install the ROS 2 packages we depend on:

```bash
sudo apt-get install -y \
  ros-jazzy-turtlebot3 \
  ros-jazzy-turtlebot3-description \
  ros-jazzy-turtlebot3-gazebo \
  ros-jazzy-ros-gz \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-ros-gz-interfaces \
  ros-jazzy-slam-toolbox \
  ros-jazzy-rviz2 \
  ros-jazzy-tf-transformations
```

Install PyTorch into the system Python 3.12 used by ROS 2 (uv or pip; pick a
CUDA build that matches your driver):

```bash
uv pip install --system --python /usr/bin/python3.12 torch
```

Clone this repo and `cd` into it:

```bash
git clone https://github.com/prakash-aryan/turtlebot3_deepRL.git
cd turtlebot3_deepRL
```

## Build
```bash
source /opt/ros/jazzy/setup.bash
colcon build --allow-overriding turtlebot3_msgs
source install/setup.bash
```

Environment to have set in each shell:

```bash
source /opt/ros/jazzy/setup.bash
source <repo>/install/setup.bash
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=1
```

## Run the demo
Four terminals (or a `tmux` session):

**1. Sim + bridges + SLAM + RViz:**
```bash
ros2 launch turtlebot3_drl_gazebo turtlebot3_drl_stage9.launch.py
```
Pass `headless:=true` to skip the Gazebo GUI and RViz (faster, used for training):
```bash
ros2 launch turtlebot3_drl_gazebo turtlebot3_drl_stage9.launch.py headless:=true
```

**2. Goal publisher:**
```bash
ros2 run turtlebot3_drl gazebo_goals
```

**3. DRL environment:**
```bash
ros2 run turtlebot3_drl environment
```

**4. Agent — pick `dqn`, `ddpg`, `td3`, or `redq`:**
```bash
ros2 run turtlebot3_drl test_agent td3  "examples/td3_0_stage9"  7400
ros2 run turtlebot3_drl test_agent ddpg "examples/ddpg_0_stage9" 8000
ros2 run turtlebot3_drl test_agent redq "examples/redq_0_stage9" 3600
```

The model name must end with the stage number — the loader reads the last
character as `stage` (e.g. `examples/td3_0_stage9` → stage 9).

## What's in the stack
| Component | Role |
|---|---|
| `gz sim` (Harmonic) | Physics + sensors + the DRL world |
| `ros_gz_bridge` (topic + service) | `/cmd_vel`, `/scan`, `/imu`, `/odom`, `/ground_truth_odom`, `/clock`, plus `SpawnEntity`, `DeleteEntity`, `ControlWorld`, `SetEntityPose` |
| `turtlebot3_drl_gazebo::ObstacleAnimator` | gz-sim system plugin that drives each moving obstacle along an SDF-defined keyframe trajectory |
| `turtlebot3_drl_gazebo/drl_burger` | Official TurtleBot3 burger SDF with lidar re-configured to 40 samples (the input size the pretrained models were trained on) |
| `slam_toolbox` (`async_slam_toolbox_node`) | Real-time occupancy map from `/scan` |
| `gt_tf_publisher` | Publishes `map → odom` computed from `/ground_truth_odom` (world frame) so RViz is always aligned even when DiffDrive's `/odom` drifts |
| `path_publisher` | Accumulates `/ground_truth_odom` into a `nav_msgs/Path` on `/robot_path` |
| `turtlebot3_drl` DRL nodes | `gazebo_goals` spawns goals and resets the robot; `environment` builds the state vector and runs the reward; `test_agent`/`train_agent` loads a PyTorch policy and issues `/cmd_vel` |

RViz (fixed frame `map`) shows: grid, TurtleBot3 model, live `/scan` with a 20 s
decay, SLAM `/map`, robot path trail, and the current goal arrow.

## Switching models / algorithms
Included pretrained examples in `src/turtlebot3_drl/model/examples/`:

| Folder | Algorithm | Episode | Observed success on stage 9 |
|---|---|---|---|
| `ddpg_0_stage9/` | DDPG | 8 000 | ~85 % |
| `td3_0_stage9/`  | TD3  | 7 400 | ~95 %+ |
| `redq_0_stage9/` | REDQ | 3 600 | ~80 % (6× fewer env-steps than TD3) |

All four algorithms (`dqn`, `ddpg`, `td3`, `redq`) share the same env +
reward + replay buffer and are selected via the first positional arg to
`test_agent` / `train_agent`.

### Training another agent from scratch
```bash
# create a new session (auto-numbered: dir becomes <host>/<algo>_N_stage_<S>)
ros2 run turtlebot3_drl train_agent td3        # replace td3 with ddpg / dqn / redq

# resume a session at a specific episode
ros2 run turtlebot3_drl train_agent td3 td3_0_stage9 500
```

Training dumps go to `src/turtlebot3_drl/model/<hostname>/<algo>_N_stage_<S>/`
with actor/critic `.pt` checkpoints every `MODEL_STORE_INTERVAL` (100) episodes
plus a running `_figure.png` plot of reward / outcome history. Hyperparameters
live in `src/turtlebot3_drl/turtlebot3_drl/common/settings.py` — each algorithm
has its own block (`POLICY_NOISE`, `REDQ_UTD_RATIO`, `DQN_ACTION_SIZE`, etc.)
that you can tune without touching the agent code. Training is GPU-accelerated
via PyTorch; run `headless:=true` on the sim launch to skip the Gazebo GUI +
RViz for a ~15–20 % speedup.

### Running on CPU (no NVIDIA GPU)

PyTorch device is picked automatically — `util.check_gpu()` returns `cuda` if
available, else `cpu`. To install the CPU-only wheel:

```bash
uv pip install --system --python /usr/bin/python3.12 \
    torch --index-url https://download.pytorch.org/whl/cpu
```

Everything works unchanged from there — testing a pretrained policy
(`test_agent td3 "examples/td3_0_stage9" 7400`) runs fine on CPU because
inference is cheap. Training is **much slower**: per-step cost is dominated
by the gradient updates, which on a modern desktop CPU run ~10–30× slower
than an RTX 4070+. For overnight runs on CPU, tune settings down:

```python
# common/settings.py — sensible CPU defaults
BATCH_SIZE              = 64        # was 128
HIDDEN_SIZE             = 256       # was 512
REDQ_ENSEMBLE_SIZE      = 5         # was 10
REDQ_UTD_RATIO          = 5         # was 20
REDQ_BATCH_SIZE         = 128       # was 512
```

DDPG / TD3 / DQN are cheaper per update and tolerate CPU training better than
REDQ (which is gradient-update bound). Expect 4–10× real-time training for
TD3/DDPG on a decent CPU vs near-real-time on a mid-range GPU.

Hyperparameters live in `src/turtlebot3_drl/turtlebot3_drl/common/settings.py`
(reward function, scan samples, arena sizes, episode length, etc.).

Only stage 9 is currently ported to gz-sim. The port pattern is
`launch/turtlebot3_drl_stage9.launch.py` + `worlds/turtlebot3_drl_stage9.world`
(with keyframes per obstacle) + optional edits to
`models/turtlebot3_drl_world/`. Copy and edit for other stages.

## Package layout
```
src/
├── turtlebot3_drl/                 # DRL algorithms, env, agent, storage, graph
│   └── turtlebot3_drl/
│       ├── drl_agent/              # DDPG, TD3, DQN
│       ├── drl_environment/        # state/reward, step service
│       ├── drl_gazebo/             # goal spawn/move, reset
│       ├── common/                 # storage, logger, settings, utilities
│       └── utility/                # path_publisher, gt_tf_publisher
├── turtlebot3_drl_gazebo/          # gz-sim world, launch, plugins, RViz, bridge yaml
│   ├── launch/
│   ├── models/                     # drl_burger, drl_world geometry
│   ├── params/                     # bridge yaml, slam_toolbox yaml
│   ├── rviz/                       # drl_nav.rviz
│   ├── src/                        # obstacle_animator.cc
│   └── worlds/                     # turtlebot3_drl_stage9.world
├── turtlebot3_msgs/                # DrlStep, Goal, RingGoal srv
└── turtlebot3_simulations/         # (COLCON_IGNORE — Classic Gazebo, not used)
```

## Troubleshooting
**mise / non-system Python breaks `colcon build`.** Build in a clean env:
```bash
env -i HOME=$HOME PATH=/usr/bin:/bin:/opt/ros/jazzy/bin bash -c \
  'source /opt/ros/jazzy/setup.bash && colcon build --allow-overriding turtlebot3_msgs'
```

**`TURTLEBOT3_MODEL` unset.** Only `burger` is used here; exported in the shell.

**Robot drifts or appears misplaced in RViz.** That happens when SLAM owns
`map → odom` and the moving obstacles throw off scan matching. This repo
publishes `map → odom` from ground truth (`gt_tf_publisher`) so the robot is
always correctly localized; SLAM just contributes the `/map` occupancy grid.

**No `/map`.** Confirm SLAM activated:
```bash
ros2 param get /slam_toolbox odom_frame   # should print "odom"
ros2 topic hz /map                        # ~2 Hz once robot moves
```

**Goal topic type mismatch.** The DRL goal topic is `/drl_goal_pose`
(`geometry_msgs/Pose`). `/goal_pose` is left for SLAM/Nav2 use with
`PoseStamped`.

## Acknowledgments
This is a ROS 2 Jazzy / Gazebo Harmonic port of **[tomasvr/turtlebot3_drlnav](https://github.com/tomasvr/turtlebot3_drlnav)** (ROS 2 Foxy + Gazebo Classic). The DRL algorithms, reward functions, stage layouts, and pretrained models
(`examples/ddpg_0_stage9`, `examples/td3_0_stage9`) are from that repo. TurtleBot3 meshes and
base SDFs come from [ROBOTIS-GIT/turtlebot3](https://github.com/ROBOTIS-GIT/turtlebot3)
and [ROBOTIS-GIT/turtlebot3_simulations](https://github.com/ROBOTIS-GIT/turtlebot3_simulations).
