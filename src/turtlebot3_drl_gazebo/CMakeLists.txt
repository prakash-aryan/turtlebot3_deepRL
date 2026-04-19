cmake_minimum_required(VERSION 3.22)
project(turtlebot3_drl_gazebo)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(gz_sim_vendor REQUIRED)
find_package(gz-sim REQUIRED)
find_package(gz_math_vendor REQUIRED)
find_package(gz-math REQUIRED)
find_package(gz_plugin_vendor REQUIRED)
find_package(gz-plugin REQUIRED)
find_package(sdformat_vendor REQUIRED)
find_package(sdformat REQUIRED)

add_library(obstacle_animator SHARED src/obstacle_animator.cc)
target_link_libraries(obstacle_animator
  gz-sim::gz-sim
  gz-math::gz-math
  gz-plugin::register
  sdformat::sdformat
)

install(TARGETS obstacle_animator
  LIBRARY DESTINATION lib/${PROJECT_NAME}
  ARCHIVE DESTINATION lib/${PROJECT_NAME}
  RUNTIME DESTINATION bin/${PROJECT_NAME}
)

install(DIRECTORY launch models worlds params rviz
  DESTINATION share/${PROJECT_NAME}/
)

ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/hooks/resource_paths.dsv.in"
)

ament_package()
