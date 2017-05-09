set(OpenCV_DIR /home/bjm/environments/opencv-3.2.0/build)
find_package(OpenCV REQUIRED COMPONENTS
core
flann
highgui
calib3d
imgproc
features2d
imgcodecs
)
if(OpenCV_FOUND)
  message(STATUS "$OpenCV_VERSION = ${OpenCV_VERSION}")
  message(STATUS "$OpenCV_LIBS = ${OpenCV_LIBS}")
else()
  message(FATAL_ERROR "Opencv not found")
endif()

set(BOOST_ROOT /home/bjm/environments/boost_1_60_0/install)
find_package(Boost REQUIRED COMPONENTS
filesystem
system
date_time)
if(Boost_FOUND)
  message(STATUS "$Boost_INCLUDE_DIR = ${Boost_INCLUDE_DIR}")
  message(STATUS "$Boost_LIBRARY_DIR = ${Boost_LIBRARY_DIR}")
  message(STATUS "$Boost_LIBRARIES = ${Boost_LIBRARIES}")
else()
  message(FATAL_ERROR "Boost not found")
endif()

include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

# 加入C++11支持
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# 指定编译64位程序
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")

# 要生成的目标文件
add_executable(opencv_alignment main.cpp my_detector.cpp util.cpp)
# 指定生成文件的输出路径
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Link your application with OpenCV libraries
target_link_libraries(opencv_alignment ${OpenCV_LIBS} ${Boost_LIBRARIES})

