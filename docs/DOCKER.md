# Docker setup

Two variants are provided:

| File | Hardware | PyTorch wheel | When to use |
|---|---|---|---|
| `docker-compose.yml` | NVIDIA GPU | CUDA 12.4 | Training (fastest); also runs inference |
| `docker-compose.cpu.yml` | CPU-only | CPU | Machines without NVIDIA, or just running pretrained policies |

## Prerequisites
* GPU variant: Docker 24+, Docker Compose v2, NVIDIA Container Toolkit, NVIDIA GPU.
* CPU variant: Docker 24+, Docker Compose v2. No GPU, no NVIDIA toolkit needed.

## Build & start — GPU
```bash
cd docker
docker compose up -d --build
```

## Build & start — CPU
```bash
cd docker
docker compose -f docker-compose.cpu.yml up -d --build
```
The CPU variant uses container name `turtlebot3-drl-cpu` and image
`turtlebot3-drl:jazzy-cpu`, so it can coexist with the GPU variant on the
same host. Inference (`test_agent` / `eval_agent` on the pretrained
policies) runs fine on CPU; training is much slower — see the README
"Running on CPU" section for tuned hyperparameters.

Gazebo's 3D viewport renders through software OpenGL (llvmpipe) inside
the container. It works, but it's slow and CPU-heavy — prefer
`headless:=true` on the sim launch when you don't need RViz / Gazebo
GUI:
```bash
ros2 launch turtlebot3_drl_gazebo turtlebot3_drl_stage9.launch.py headless:=true
```

## Browser (noVNC)
```
http://<host>:6080/
```

## Workspace build
For the GPU variant:
```bash
docker compose exec drl bash
source /opt/ros/jazzy/setup.bash
colcon build --allow-overriding turtlebot3_msgs
exit
```
For the CPU variant, add `-f docker-compose.cpu.yml` to every
`docker compose` command:
```bash
docker compose -f docker-compose.cpu.yml exec drl bash
source /opt/ros/jazzy/setup.bash
colcon build --allow-overriding turtlebot3_msgs
exit
```

## Run the demo (4 terminals)
In each terminal (GPU variant shown; for CPU prepend
`-f docker-compose.cpu.yml`):
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
GPU:
```bash
docker compose down
```
CPU:
```bash
docker compose -f docker-compose.cpu.yml down
```