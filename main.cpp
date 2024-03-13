#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>
#include "data_processing/data_processing.hpp"
#include <math.h>
#include "matplotlibcpp.h"

using namespace cv;
using namespace std;
namespace plt = matplotlibcpp;


Mat img; // Исходное изображение
vector<pair<double, double>> crossCoords; // Вектор для хранения координат крестиков

// Функция для рисования всех крестиков на изображении и отображения результата
void updateImage() {

    Mat imgCopy = img.clone();

    for (auto &coord: crossCoords) {
        int x = int(coord.first);
        int y = int(coord.second);

        int size = 5;
        int thickness = 3;
        Scalar color = Scalar(0, 0, 255);

        line(imgCopy, Point(x - size, y), Point(x + size, y), color, thickness);
        line(imgCopy, Point(x, y - size), Point(x, y + size), color, thickness);
    }

    imshow("viewing window", imgCopy);
}

bool dragging = false; // Флаг для коректной обработки перетаскивания картинки внутри окна

void CallBackFunc(int event, int x, int y, int flags, void *userdata) {
    static Point prevPoint; // Предыдущая точка для сравнения

    //Логика работы этой функции позволяет не реагировать на нажатия, если после нажатия произошло перемещение курсора
    if (event == EVENT_LBUTTONDOWN) {
        //кнопка была нажата
        dragging = true;
        prevPoint = Point(x, y);
    } else if (event == EVENT_MOUSEMOVE && dragging) {
        //после нажатия кнопки произошло перетаскивание, нажатие теперь не актуально
        if (norm(Point(x, y) - prevPoint) > 0) {
            dragging = false;
        }
    } else if (event == EVENT_LBUTTONUP) {
        //кнопка была отжата, и если перемещений не было, то рисуется крестик
        if (dragging) {
            if (crossCoords.size() < 2) {
                crossCoords.push_back(make_pair(x, y));
            }
            updateImage();
        }
        dragging = true;
    }
}

int main(int argc, char **argv) {

    if (argc != 2) {
        cout << "Использование: " << argv[0] << " <путь_к_изображению>" << endl;
        return -1;
    }
    img = imread(argv[1]);
    if (img.empty()) {
        cout << "Ошибка загрузки изображения" << endl;
        return -1;
    }

    namedWindow("viewing window", WINDOW_NORMAL);
    setMouseCallback("viewing window", CallBackFunc, NULL);
    updateImage();

    while (true) {
        int key = waitKey(0);

        switch(key){

            //удаление последнего креста из добавленых
            case 8:
            case 127: {
                if (!crossCoords.empty()) {
                    crossCoords.pop_back();
                    updateImage(); // Обновляем изображение
                }
                break;
            }

            //выход из программы
            case 27: {
                return 0;
            }

            //уточнение координат
            case 'f': {
                pinPoint(argv[1], crossCoords);
                updateImage();
            }

            //Обработка изображений
            case 13:
            case 10: {
                process_data(argv[1], crossCoords);

            }
        }
    }

    return 0;
}