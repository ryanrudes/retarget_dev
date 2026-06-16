cmake_minimum_required(VERSION 3.16)


project(vicon_bridge)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

################
# DEPENDENCIES #
################

find_package(Boost REQUIRED COMPONENTS thread date_time chrono)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(diagnostic_updater REQUIRED)

get_filename_component(_ros_env_root "${ament_cmake_DIR}/../../.." ABSOLUTE)
set(_vicon_bridge_python "${_ros_env_root}/bin/python3")
set(Python3_EXECUTABLE "${_vicon_bridge_python}" CACHE FILEPATH "" FORCE)
set(Python_EXECUTABLE "${_vicon_bridge_python}" CACHE FILEPATH "" FORCE)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Marker.msg"
  "msg/Markers.msg"
  DEPENDENCIES geometry_msgs std_msgs
)


include_directories(
	${Boost_INCLUDE_DIRS}
	vicon_sdk/DataStream
	vicon_sdk/DataStream/ViconDataStreamSDK_CPP
	vicon_sdk
	include
	)
#############
# VICON SDK #
#############

# Compile ViconSDK from source (exclude bundled SDK test mains).
file(GLOB_RECURSE vicon_sdk_files "${CMAKE_CURRENT_SOURCE_DIR}/vicon_sdk/**/**.cpp")
list(FILTER vicon_sdk_files EXCLUDE REGEX ".*/ViconDataStreamSDK_.*Test/.*")
add_library(vicon_sdk STATIC ${vicon_sdk_files})
target_link_libraries(vicon_sdk PUBLIC ${Boost_LIBRARIES})


###########################
# NON-ROS TEST EXECUTABLE #
###########################
add_executable(vicon_test src/vicon_test.cpp)
target_link_libraries(vicon_test vicon_sdk)

add_executable(vicon_test_fps src/vicon_test_fps.cpp)
target_link_libraries(vicon_test_fps vicon_sdk)

##################
# ROS EXECUTABLE #
##################

# rcutils exports -latomic for Linux; macOS libc++ already provides atomics.
if(APPLE)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/atomic_stub.c" "")
  add_library(atomic STATIC "${CMAKE_CURRENT_BINARY_DIR}/atomic_stub.c")
  set_target_properties(atomic PROPERTIES OUTPUT_NAME atomic)
endif()

add_executable(vicon_bridge_node src/vicon_bridge.cpp)
set_target_properties(vicon_bridge_node PROPERTIES OUTPUT_NAME vicon_bridge)
ament_target_dependencies(vicon_bridge_node
	rclcpp
	tf2
	tf2_ros
	geometry_msgs
  diagnostic_updater
)
target_link_libraries(vicon_bridge_node vicon_sdk ${Boost_LIBRARIES})
if(APPLE)
  target_link_directories(vicon_bridge_node PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endif()

add_executable(vicon_bridge_retiming src/vicon_bridge_retiming.cpp)
ament_target_dependencies(vicon_bridge_retiming
	rclcpp
	tf2
	tf2_ros
	geometry_msgs
)
target_link_libraries(vicon_bridge_retiming vicon_sdk ${Boost_LIBRARIES})
if(APPLE)
  target_link_directories(vicon_bridge_retiming PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
endif()

install(TARGETS
	vicon_bridge_node
       	vicon_bridge_retiming
	DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
	launch
	DESTINATION share/${PROJECT_NAME}/
)




ament_export_dependencies(rosidl_default_runtime)
ament_package()
