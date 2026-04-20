import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


STAGE = '9'
WORLD_NAME = 'drl_stage9'


def generate_launch_description():
    drl_gz_share = get_package_share_directory('turtlebot3_drl_gazebo')
    tb3_gz_share = get_package_share_directory('turtlebot3_gazebo')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    x_pose = LaunchConfiguration('x_pose', default='0.0')
    y_pose = LaunchConfiguration('y_pose', default='0.0')
    headless = LaunchConfiguration('headless', default='false')

    world = os.path.join(drl_gz_share, 'worlds', 'turtlebot3_drl_stage9.world')

    tb3_model = 'burger'
    urdf_path = os.path.join(drl_gz_share, 'models', 'drl_burger', 'model.sdf')

    bridge_yaml = os.path.join(drl_gz_share, 'params', 'drl_bridge.yaml')

    set_drl_models = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(drl_gz_share, 'models'),
    )
    set_tb3_models = AppendEnvironmentVariable(
        'GZ_SIM_RESOURCE_PATH',
        os.path.join(tb3_gz_share, 'models'),
    )
    set_plugin_path = AppendEnvironmentVariable(
        'GZ_SIM_SYSTEM_PLUGIN_PATH',
        os.path.join(drl_gz_share, '..', '..', 'lib', 'turtlebot3_drl_gazebo'),
    )

    write_stage = ExecuteProcess(
        cmd=['bash', '-c', f'echo {STAGE} > /tmp/drlnav_current_stage.txt'],
        output='screen',
    )

    gzserver_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': ['-r -s -v2 ', world]}.items(),
    )
    gzclient_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-g -v2 '}.items(),
        condition=UnlessCondition(headless),
    )

    robot_state_publisher_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(tb3_gz_share, 'launch', 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items(),
    )

    spawn_turtlebot_cmd = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', tb3_model,
            '-file', urdf_path,
            '-x', x_pose,
            '-y', y_pose,
            '-z', '0.01',
        ],
        output='screen',
    )

    bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '--ros-args',
            '-p',
            f'config_file:={bridge_yaml}',
        ],
        output='screen',
    )

    gz_service_bridge_args = [
        f'/world/{WORLD_NAME}/control@ros_gz_interfaces/srv/ControlWorld',
        f'/world/{WORLD_NAME}/create@ros_gz_interfaces/srv/SpawnEntity',
        f'/world/{WORLD_NAME}/remove@ros_gz_interfaces/srv/DeleteEntity',
        f'/world/{WORLD_NAME}/set_pose@ros_gz_interfaces/srv/SetEntityPose',
    ]
    service_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=gz_service_bridge_args,
        output='screen',
    )

    path_publisher_cmd = Node(
        package='turtlebot3_drl',
        executable='path_publisher',
        output='screen',
    )

    gt_tf_cmd = Node(
        package='turtlebot3_drl',
        executable='gt_tf_publisher',
        output='screen',
    )

    slam_toolbox_share = get_package_share_directory('slam_toolbox')
    slam_params = os.path.join(drl_gz_share, 'params', 'slam_toolbox.yaml')
    slam_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(slam_toolbox_share, 'launch', 'online_async_launch.py')
        ),
        launch_arguments={
            'slam_params_file': slam_params,
            'use_sim_time': 'true',
        }.items(),
    )

    nav2_bringup_share = get_package_share_directory('nav2_bringup')
    nav2_params = os.path.join(drl_gz_share, 'params', 'nav2_params.yaml')
    nav2_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_share, 'launch', 'navigation_launch.py')
        ),
        launch_arguments={
            'use_sim_time': 'true',
            'params_file': nav2_params,
            'autostart': 'true',
            'use_composition': 'False',
        }.items(),
    )

    rviz_config = os.path.join(drl_gz_share, 'rviz', 'drl_nav.rviz')
    rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        output='screen',
        condition=UnlessCondition(headless),
    )

    return LaunchDescription([
        DeclareLaunchArgument('headless', default_value='false',
                              description='Skip gzclient + RViz (faster for training)'),
        set_drl_models,
        set_tb3_models,
        set_plugin_path,
        write_stage,
        gzserver_cmd,
        gzclient_cmd,
        robot_state_publisher_cmd,
        spawn_turtlebot_cmd,
        bridge_cmd,
        service_bridge_cmd,
        path_publisher_cmd,
        gt_tf_cmd,
        slam_cmd,
        rviz_cmd,
    ])
