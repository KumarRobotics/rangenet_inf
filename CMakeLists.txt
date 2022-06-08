cmake_minimum_required(VERSION 3.16)
project(scan2shape)

set(CMAKE_CXX_STANDARD 17)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  geometry_msgs
  pcl_conversions
  pcl_ros
  roscpp
)

find_package(Boost REQUIRED system serialization)
find_package(GTSAM REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(glog REQUIRED)

catkin_package(INCLUDE_DIRS src LIBRARIES ${PROJECT_NAME})

include_directories(${PROJECT_NAME} PUBLIC ${catkin_INCLUDE_DIRS} src)

add_executable(data_loader src/data_loader.cpp)
target_link_libraries(data_loader PUBLIC ${catkin_LIBRARIES})