cmake_minimum_required(VERSION 2.8)
project( omni_calib )
find_package( OpenCV REQUIRED )
add_executable( omni_calib omni_calibration.cpp )
add_executable( imagelist_creator imagelist_creator.cpp )
target_link_libraries( omni_calib ${OpenCV_LIBS} )
target_link_libraries( imagelist_creator ${OpenCV_LIBS} )