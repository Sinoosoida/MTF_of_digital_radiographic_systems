cmake_minimum_required(VERSION 3.26)
project(cv)

set(CMAKE_CXX_STANDARD 17) # Указание использовать стандарт C++17
set(CMAKE_CXX_STANDARD_REQUIRED True) # Стандарт C++17 обязателен
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Находим пакеты
find_package(Qt5Widgets REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# Добавляем путь к заголовочным файлам Python, NumPy и matplotlibcpp
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)

# Если у вас есть файлы исходного кода для data_processing, добавьте их здесь
# Например, добавление data_processing как статической библиотеки
add_library(data_processing STATIC include/data_processing/data_processing.hpp)

# Добавляем исполняемый файл
add_executable(cv main.cpp)

# Связываем библиотеки с исполняемым файлом
target_link_libraries(cv Qt5::Widgets ${OpenCV_LIBS} Python3::Python Python3::NumPy data_processing)
