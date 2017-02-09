cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
MESSAGE("FALGS: ${CMAKE_CXX_FLAGS}")

project(nan)

find_package(CURL)
find_package(JPEG)
find_package(PNG)

include_directories(
/misc/lmbraid11/tananaed/tools/curl-7.52.1/inst/include
#/misc/lmbraid11/tananaed/tools/libpng-1.6.28/inst/include
/misc/lmbraid11/tananaed/tools/png++/include)
add_executable(nan 
	nan.cpp)

target_link_libraries(nan ${CURL_LIBRARIES} ${JPEG_LIBRARIES} ${PNG_LIBRARIES})

