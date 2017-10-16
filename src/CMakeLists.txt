cmake_minimum_required(VERSION 2.8)

#project(opencv_alignment CXX)

set(BOOST_INCLUDE_DIR /home/bjm/environments/boost_1_60_0/install/include)
set(BOOST_LIBRARY_DIR /home/bjm/environments/boost_1_60_0/install/lib)
set(BOOST_LIBS boost_date_time boost_filesystem boost_system)

set(OPENCV_INCLUDE_DIR /home/bjm/environments/opencv-3.2.0/install/include)
set(OPENCV_LIBRARY_DIR /home/bjm/environments/opencv-3.2.0/install/lib)
set(OPENCV_LIBS opencv_calib3d opencv_features2d opencv_highgui opencv_imgproc opencv_imgcodecs opencv_core)

include_directories(${BOOST_INCLUDE_DIR} ${OPENCV_INCLUDE_DIR})
link_directories(${BOOST_LIBRARY_DIR} ${OPENCV_LIBRARY_DIR})

# 加入C++11支持
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
# 指定编译64位程序
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64")

# 要生成的目标文件
add_executable(opencv_alignment main.cpp my_detector.cpp util.cpp)
# 指定生成文件的输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)

# Link your application with OpenCV libraries
target_link_libraries(opencv_alignment ${OPENCV_LIBS} ${BOOST_LIBS})

