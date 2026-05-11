# https://hub.docker.com/layers/library/ros/jazzy-ros-base/images/sha256-f652c502c971653afc2ed9b9e5929643df644e6125b6ee68349408b6cc2f64cd?context=explore
FROM ros:jazzy-ros-base
ENV DEBIAN_FRONTEND=noninteractive

# Layer 1: ROS Jazzy + Gazebo Harmonic + TurtleBot3 + SLAM, build tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-turtlebot3 \
        ros-jazzy-turtlebot3-description \
        ros-jazzy-turtlebot3-gazebo \
        ros-jazzy-ros-gz \
        ros-jazzy-ros-gz-bridge \
        ros-jazzy-ros-gz-sim \
        ros-jazzy-ros-gz-interfaces \
        ros-jazzy-slam-toolbox \
        ros-jazzy-rviz2 \
        ros-jazzy-tf-transformations \
        python3-transforms3d \
        python3-colcon-common-extensions \
        python3-rosdep \
        python3-pip \
        git tmux nano \
    && rm -rf /var/lib/apt/lists/*

# Layer 2: headless X stack + noVNC. Xvnc renders RViz/gz-sim into a virtual
# framebuffer, fluxbox gives window decorations, websockify serves noVNC over
# HTTP so a host browser can connect at http://<host>:6040/vnc.html.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tigervnc-standalone-server \
        tigervnc-common \
        novnc \
        websockify \
        fluxbox \
        xterm \
        dbus-x11 \
        mesa-utils \
        libgl1-mesa-dri \
    && ln -sf /usr/share/novnc/vnc.html /usr/share/novnc/index.html \
    && rm -rf /var/lib/apt/lists/*

# Layer 3: PyTorch (CUDA 12.4 wheels) + colcon override-check (not in apt on
# Jazzy; needed for --allow-overriding turtlebot3_msgs).
RUN pip3 install --no-cache-dir --break-system-packages \
        --index-url https://download.pytorch.org/whl/cu124 \
        torch \
    && pip3 install --no-cache-dir --break-system-packages \
        colcon-override-check

# Layer 4: env, bashrc, entrypoint.
ENV TURTLEBOT3_MODEL=burger \
    ROS_DOMAIN_ID=1 \
    DISPLAY=:1

WORKDIR /root/turtlebot3_deepRL

RUN cat >> /root/.bashrc <<'EOF'

source /opt/ros/jazzy/setup.bash
[ -f /root/turtlebot3_deepRL/install/setup.bash ] && source /root/turtlebot3_deepRL/install/setup.bash
export TURTLEBOT3_MODEL=burger
export ROS_DOMAIN_ID=1
export DRLNAV_BASE_PATH=/root/turtlebot3_deepRL
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/root/turtlebot3_deepRL/src/turtlebot3_drl_gazebo/models
EOF

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 6040
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["tail", "-f", "/dev/null"]
