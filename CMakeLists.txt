cmake_minimum_required(VERSION 3.16)
project(FEM VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

if(MSVC)
    add_compile_options(/W4 /WX-)
else()
    add_compile_options(-Wall -Wextra -Werror)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(examples)