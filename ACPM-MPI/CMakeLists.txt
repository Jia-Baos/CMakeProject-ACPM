# CMakeList.txt: main 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.27)

# Enable Hot Reload for MSVC compilers if supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project(acpm-mpi)

message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Project Dir: ${CMAKE_CURRENT_SOURCE_DIR}")

# 将源代码添加到此项目的可执行文件。
aux_source_directory(. main_files)
add_executable(${PROJECT_NAME} ${main_files})

# TODO: 如有需要，请添加测试并安装目标。

set(Lib_Dir "D:/opencv-4.6.0-contrib/build")

# 链接库头文件路径
target_include_directories(${PROJECT_NAME}
        PUBLIC ${Lib_Dir}/include)
target_include_directories(${PROJECT_NAME}
        PUBLIC ${Lib_Dir}/include/opencv2)

# 链接库文件路径
target_link_directories(${PROJECT_NAME}
        PUBLIC ${Lib_Dir}/x64/vc17/lib)

# 链接库文件名称
set(Lib_Debug "opencv_world460d")
set(Lib_Release "opencv_world460")
target_link_libraries(${PROJECT_NAME}
        PUBLIC
        debug ${Lib_Debug}
		optimized ${Lib_Release}
)
