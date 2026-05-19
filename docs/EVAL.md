# Evaluation suite

`eval_agent` runs a trained policy through a fixed list of `(start, goal)`
scenarios. The obstacle animator's phase resets at the start of each
scenario, so every algorithm sees the same world from t=0. Default suite
is 20 scenarios (5 easy / 8 medium / 5 hard / 2 edge) in
[scenarios.yaml](../src/turtlebot3_drl/turtlebot3_drl/eval/scenarios.yaml).

## Prerequisites
Built workspace + a trained policy. Pretrained examples ship in
`src/turtlebot3_drl/model/examples/{td3,ddpg,redq}_0_stage9`.

## Run (4 terminals)
Source the env in each:
```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=1
export DRLNAV_BASE_PATH=$(pwd)
```

Then one per terminal:
```bash
# T1
ros2 launch turtlebot3_drl_gazebo turtlebot3_drl_stage9.launch.py   # headless:=true to skip GUI
# T2
ros2 run turtlebot3_drl gazebo_goals
# T3
ros2 run turtlebot3_drl environment
# T4 — pick one
ros2 run turtlebot3_drl eval_agent td3  examples/td3_0_stage9  7400
ros2 run turtlebot3_drl eval_agent ddpg examples/ddpg_0_stage9 8000
ros2 run turtlebot3_drl eval_agent redq examples/redq_0_stage9 3600
```

In Docker, run the same commands inside `docker compose exec drl bash`
(prepend `-f docker-compose.cpu.yml` for the CPU image — see
[DOCKER.md](DOCKER.md)).

## Output
| File | Contents |
|---|---|
| `model/<session>/_eval_stage9_eps<E>_<datetime>.csv` | one row per scenario |
| `model/__eval_comparison.csv` | one summary row per run, appended across algorithms |

## Visualization (RViz)
* 🟢 green sphere — start
* ⚪ white sphere — goal
* text at the bottom of the orbit view: `ALGO  N/total  duration  STATUS`
  (yellow=running, green=SUCCESS, red=COLL_WALL, orange=COLL_OBST,
  gray=TIMEOUT, purple=TUMBLE)

## Outcome codes
| Code | Meaning |
|---|---|
| 1 | SUCCESS — within 0.20 m of goal |
| 2 | COLL_WALL |
| 3 | COLL_OBST |
| 4 | TIMEOUT — 50 sim seconds |
| 5 | TUMBLE |

## Caveats
* `eval_agent` latches `/eval_mode_active=True` so `gazebo_goals` stops
  emitting random goals during the run. Restart that node before going
  back to `test_agent`.
* SLAM map briefly resets each scenario (clock rewinds for obstacles) —
  does not affect score.
* Session name must end with the stage number (`..._stage9` → stage 9).

## Add a scenario
Edit [scenarios.yaml](../src/turtlebot3_drl/turtlebot3_drl/eval/scenarios.yaml):
```yaml
- id: S21
  tag: my_scenario
  difficulty: hard
  start: {x:  1.0, y: -0.5, yaw: 0.0}
  goal:  {x: -1.5, y:  1.2}
```
Rebuild with `colcon build --packages-select turtlebot3_drl`.
