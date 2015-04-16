# CMakeLists files in this project can	
# refer to the root source directory of the project as ${MCL_CPP_SOURCE_DIR} and
# to the root binary directory of the project as ${MCL_CPP_BINARY_DIR}.
cmake_minimum_required (VERSION 2.8.11)

project (MCL_CPP)
 
# Linking statically to boost. If your project requires Dynamic linking, change here
set(Boost_USE_STATIC_LIBS ON) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF)
set(BOOST_ROOT $ENV{BOOST_ROOT} ) 
FIND_PACKAGE(Boost 1.57.0 COMPONENTS system thread date_time chrono graph unit_test_framework)
if(Boost_FOUND)
	SET(MCL_INCLUDE_DIRS ${Boost_INCLUDE_DIRS})
	add_definitions(-DBOOST_LIB_DIAGNOSTIC)
	MESSAGE(STATUS "Found Boost_Libraries=" ${Boost_LIBRARIES})
endif()
 
## EIGEN
SET(MCL_INCLUDE_DIRS ${MCL_INCLUDE_DIRS} ${EIGEN_ROOT}/include/Eigen3 ${MCL_CPP_SOURCE_DIR})
SET(MCL_STATIC_LINK_LIBS ${MCL_STATIC_LINK_LIBS} ${Boost_LIBRARIES})

INCLUDE_DIRECTORIES(${MCL_INCLUDE_DIRS}) 

ADD_LIBRARY(mcl INTERFACE)
target_include_directories(mcl INTERFACE $<BUILD_INTERFACE:${MCL_CPP_SOURCE_DIR}/mcl> )

#FILE(GLOB srcs ${MCL_CPP_SOURCE_DIR}/mcl ${MCL_CPP_SOURCE_DIR}/mcl/*.cpp ${MCL_CPP_SOURCE_DIR}/mcl/*.hpp)
#ADD_LIBRARY(mcl INTERFACE ${srcs})
	
ADD_EXECUTABLE(Test ${MCL_CPP_SOURCE_DIR}/Test/main.cpp)
TARGET_LINK_LIBRARIES(Test ${MCL_STATIC_LINK_LIBS})

