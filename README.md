# Данный проект реализует метод измерения ЧКХ цифровых рентгенографических систем с предварительной дискретизацией

Алгоритмы, их параметры, и способы их использования взяты из этих статей:
 - A method for measuring the presampled MTF of digital radiographic
   systems using an edge test device 
 - Conditioning data for calculation of the modulation transfer function
 - Assessment of volumetric noise and resolution performance for linear
   and nonlinear CT reconstruction methods
 - Performance Evaluation
   of Computed Tomography Systems

Алгоритм определения характеристик рентгенографических систем выглядит следующим образом.
 - Предарительное определение границы разделов двух сред.
 - Уточнения местоположения этой границы.
 - Отображени пикселей на ось, перпендикулярную разделу двух сред, получение сырой ESF
 - Усреднение значений ESF по бинам.
 - Усреднение с помощью оконного Гуацсово-взвешенного полиномиального фильтра
 - Диференцирование - получение LSF.
 - Фильтрация LSF c помощью окна Ханна.
 - Разложение фурье - получение MTF.
 - Нахождение параметров разложение фурье - получение MTF0.5 и MTF0.1

Реализация всех алгоритмов находится в библиотеке [data_processing.hpp](include%2Fdata_processing%2Fdata_processing.hpp)
Реализация оконного интерфейса находится в [main.cpp](main.cpp)
CMake проекта [CMakeLists.txt](CMakeLists.txt)

Сборка проекта
```bash
./build.sh
```
Запуск проекта
```bash
./build/cv ./Data/img.png
```
При запуске проекта, в качестке аргумента, необходимо передать путь до фотографии, которая будет обработана. 
Все помежуточные этапы обработки автоматически сохранятся в ту же директорию, где и лежить фотография.

Управление графическим интерфейсом:
 - Нажатие мышкой без передвижения - установка точки. Две точки задают линию.
 - f (fit) автоматически переместить нарисованый отпрезок на ближайшую область раздела сред.
 - Enter - обработать изображение (две точки уже должны быть поставлены)
 - Esc - выход из графического интерфейса.