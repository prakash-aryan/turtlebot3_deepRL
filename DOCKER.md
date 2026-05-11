# Docker setup

## Prerequisites
Docker 24+, Docker Compose v2, NVIDIA Container Toolkit, NVIDIA GPU.

## Build & start
```bash
cd docker
docker compose up -d --build
```

## Browser (noVNC)
```
http://<host>:6080/
```

## First-time workspace build
```bash
docker compose exec drl bash
source /opt/ros/jazzy/setup.bash
colcon build --allow-overriding turtlebot3_msgs
exit
```

## Run the demo (4 terminals)
In each terminal:
```bash
docker compose exec drl bash
source install/setup.bash
```

Then one per terminal, ~3 s apart:
```bash
# T1
ros2 launch turtlebot3_drl_gazebo turtlebot3_drl_stage9.launch.py   # headless:=true to skip GUI
# T2
ros2 run turtlebot3_drl gazebo_goals
# T3
ros2 run turtlebot3_drl environment
# T4
ros2 run turtlebot3_drl test_agent td3 "examples/td3_0_stage9" 7400
```

## Tear down
```bash
docker compose down
```

## Troubleshooting
| Symptom | Fix |
|---|---|
| `Bind for 0.0.0.0:6080 failed` | `docker stop <other-container-on-6080>` (often `px4_sitl`) |
| `Temporary failure resolving` | Confirm `dns:` block in `docker-compose.yml` |
| `TypeError: NoneType + str` in `gazebo_goals` | `export DRLNAV_BASE_PATH=/root/turtlebot3_deepRL` |
| `colcon: unrecognized arguments: --allow-overriding` | `pip3 install --break-system-packages colcon-override-check` |
| `torch.cuda.is_available() == False` | Check NVIDIA Container Toolkit on host |
| Gazebo slow/black in browser | Expected (software GL); use `headless:=true` |
