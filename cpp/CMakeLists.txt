cmake_minimum_required(VERSION 3.16)
project(embedding_model CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

add_executable(embedding ${SOURCES})
target_include_directories(embedding PRIVATE include)
