cmake_minimum_required(VERSION 3.0.2)
project(realsense_camera)
add_compile_options(-std=c++14)
add_definitions(-w)


option(BUILD_WITH_OPENMP "Use OpenMP" OFF)
option(SET_USER_BREAK_AT_STARTUP "Set user wait point in startup (for debug)" OFF)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)


if(BUILD_WITH_OPENMP)
    find_package(OpenMP)
    if(NOT OpenMP_FOUND)
        message(FATAL_ERROR "\n\n OpenMP is missing!\n\n")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endif()
endif()

if(SET_USER_BREAK_AT_STARTUP)
	message("GOT FLAG IN CmakeLists.txt")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBPDEBUG")
endif()

if (WIN32)
find_package(realsense2 CONFIG REQUIRED)
else()
find_package(realsense2 2.50.0)
endif()

if(NOT realsense2_FOUND)
    message(FATAL_ERROR "\n\n Intel RealSense SDK 2.0 is missing, please install it from https://github.com/IntelRealSense/librealsense/releases\n\n")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if (${uppercase_CMAKE_BUILD_TYPE} STREQUAL "RELEASE")
    message(STATUS "Create Release Build.")
    set(CMAKE_CXX_FLAGS "-O2 ${CMAKE_CXX_FLAGS}")
else()
    message(STATUS "Create Debug Build.")
endif()


if (WIN32)
else()
set(CMAKE_CXX_FLAGS "-fPIE -fPIC -std=c++11 -D_FORTIFY_SOURCE=2 -fstack-protector -Wformat -Wformat-security -Wall ${CMAKE_CXX_FLAGS}")
endif()





set(CMAKE_NO_SYSTEM_FROM_IMPORTED true)

include_directories(
    include
    ${realsense2_INCLUDE_DIR}
    )

if(WIN32)
set_target_properties(${realsense2_LIBRARY} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
target_link_libraries(${PROJECT_NAME}
    realsense2::realsense2 
    realsense2::realsense-file
    )
endif()


# ros
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  )
  
catkin_package()
include_directories(
 ${catkin_INCLUDE_DIRS})
#  ----end-ros----------

add_executable(list_device_camera
src/list_device_camera.cpp
)
target_link_libraries(list_device_camera
realsense2
)


