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

## Workspace build
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

Then one per terminal:
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

## Shut down
```bash
docker compose down
```