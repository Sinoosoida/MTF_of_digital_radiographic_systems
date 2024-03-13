#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <stdio.h>      /* printf */
#include <math.h>       /* atan */
#include "matplotlibcpp.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility> // Для std::pair
#include <string>

using namespace cv;
using namespace std;
namespace plt = matplotlibcpp;
namespace fs = std::filesystem;


const std::string LOG_FOLDER = "processing_log";
const bool LOG_PLOTS = true;
const bool LOG_DATA = true;

//Вспомогательный код для геометрии

class Line {
public:
    void Update() {
        if (a < 0) {
            a = -a;
            b = -b;
            c = -c;
        }
        double ln = sqrt(a * a + b * b);
        a = a / ln;
        b = b / ln;
        c = c / ln;
    }

    Line(double a, double b, double c) : a(a), b(b), c(c) {
        Update();
    }

    Line(const std::pair<double, double> &p1,
         const std::pair<double, double> &p2) {

        double x1 = p1.first;
        double y1 = p1.second;

        double x2 = p2.first;
        double y2 = p2.second;


        a = y1 - y2;
        b = x2 - x1;
        c = -a * x1 - b * y1;
        Update();
    }

    double getA() const { return a; }

    double getB() const { return b; }

    double getC() const { return c; }

    std::pair<double, double> intersection(const Line &other) const {
        double d = a * other.b - other.a * b;
        if (d == 0) {
            throw "Lines are parallel";
        }

        double x = (b * other.c - other.b * c) / d;
        double y = (other.a * c - a * other.c) / d;

        return {x, y};
    }

    Line getPerpendicularLine() {

        double a_perp = b;
        double b_perp = -a;

        return {a_perp, b_perp, 0};
    }

    Line getParallelLineThroughPoint(const std::pair<double, double> &point) {

        double x0 = point.first;
        double y0 = point.second;

        // вычисляем свободный член из условия прохождения через точку
        double c = -a * x0 - b * y0;
        return {a, b, c};
    }

    void print() {
        std::cout << "x*" << a << "+y*" << b << "+" << c << "=0" << std::endl;
    }

    double dist(const std::pair<double, double> &point) {
        double numerator = abs(a * point.first + b * point.second + c);
        double denominator = sqrt(a * a + b * b);
        return numerator / denominator;
    }

    double orientedDist(const std::pair<double, double> &point) {
        double numerator = a * point.first + b * point.second + c;
        double denominator = sqrt(a * a + b * b);
        return numerator / denominator;
    }

private:
    double a;
    double b;
    double c;
};

//по двум точкам (задающим отрезок) и пути к картинке, находит все пиксели, проекции которых на линию, задаваемую отрезком, лежат в отрезке.
//Положение пикселей относительно линии и яркость пикселей передаются ввиде выходного массива.

vector<pair<double, double>>
calculateDistAndBrightness(std::string file_path, const std::vector<pair<double, double>> &points) {
    Mat img = imread(file_path);

    vector<pair<double, double>> distAndBrightness;

    auto p1 = points[0];
    auto p2 = points[1];

    Line line(p1, p2);
    auto line_1 = line.getPerpendicularLine().getParallelLineThroughPoint(p1);
    auto line_2 = line.getPerpendicularLine().getParallelLineThroughPoint(p2);

    std::pair<double, double> middle_point = std::make_pair(
            (p1.first + p2.first) / 2.0, (p1.second + p2.second) / 2.0);
    double middle_point_sign = line_1.orientedDist(middle_point) * line_2.orientedDist(middle_point);

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            std::pair<double, double> point = {x, y};
            if (line_1.orientedDist(point) * line_2.orientedDist(point) * middle_point_sign > 0) {
                double distance = line.orientedDist(point);
                double brightness = 0;
                if (img.channels() == 1) {
                    brightness = img.at<uchar>(y, x);
                } else if (img.channels() == 3) {
                    Vec3b intensity = img.at<Vec3b>(y, x);
                    brightness = (intensity[0] + intensity[1] + intensity[2]) / 3.0;
                }
                distAndBrightness.push_back({distance, brightness});
            }
        }
    }
    std::sort(distAndBrightness.begin(), distAndBrightness.end());
    return distAndBrightness;
}

//бинаризирует точки

void binarize(std::vector<std::pair<double, double>> &distAndBrightness, double binWidth = 0.1) {
    // Сортировка вектора по x
    std::sort(distAndBrightness.begin(), distAndBrightness.end());

    std::vector<std::pair<double, double>> binned;

    //дполнительные функции для работы с бинами
    size_t count = 0;
    double sum = 0;
    double bin_x = 0;

    auto Initialize_bin = [&](double x) {
        if (x != bin_x) {
            bin_x = x;
            sum = 0;
            count = 0;
        }
    };
    auto Add_point = [&](std::pair<double, double> point) {
        if ((bin_x - binWidth / 2 < point.first) && (point.first <= bin_x + binWidth / 2) && (count != -1)) {
            count += 1;
            sum += point.second;
            return true;
        } else {
            return false;
        }
    };
    auto Save_bin = [&]() {
        if (count == 0) return false;
        binned.emplace_back(bin_x, sum / double(count));
        count = 0;
        return true;
    };

    //Обрабатываем нулевой бин, находя так же и точки, с которых начнётся обход в каждую из сторон
    Initialize_bin(0);
    int negative_bins_point = -1;
    int positive_bins_point = -1;
    for (int i = 0; i < distAndBrightness.size(); i++) {
        if (Add_point(distAndBrightness[i])) {
            positive_bins_point = i;
        } else {
            if (positive_bins_point != -1) {
                positive_bins_point++;
                break;
            } else {
                negative_bins_point = i;
            }
        }
    }

    if (!Save_bin()) {
        std::cerr << "No data in 0 bin, impossible to operate";
        distAndBrightness = std::move(binned);
        return;
    }

    //обрабатываем положительные бины
    int bin_num = 1;
    Initialize_bin(bin_num * binWidth);
    for (int i = positive_bins_point; i < distAndBrightness.size(); i++) {//проходимся последовательно по точкам массива
        if (!Add_point(distAndBrightness[i])) {//Если в текущий бин точка не вошла
            if (!Save_bin()) {
                break;
            }
            bin_num++;//Должна войти в следующий
            Initialize_bin(bin_num * binWidth);
            if (!Add_point(distAndBrightness[i])) {//Если и в следующий не вошла
                break;//прерываем проход
            }
        }
    }

    //обрабатываем отрицательные бины
    bin_num = -1;
    Initialize_bin(bin_num * binWidth);
    for (int i = negative_bins_point; i >= 0; i--) {//проходимся последовательно по точкам массива
        if (!Add_point(distAndBrightness[i])) {//Если в текущий бин точка не вошла
            if (!Save_bin()) {
                break;
            }
            bin_num--;//Должна войти в следующий
            Initialize_bin(bin_num * binWidth);
            if (!Add_point(distAndBrightness[i])) {//Если и в следующий не вошла
                break;//прерываем проход
            }
        }
    }

    // Сортируем
    std::sort(binned.begin(), binned.end());
    // Заменяем исходный вектор усреднёнными значениями
    distAndBrightness = std::move(binned);
}

//спомогательные функции для оконного полиномиального взвешеного фильтра
std::vector<double> gaussJordan(std::vector<std::vector<double>> A, std::vector<double> b) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        A[i].push_back(b[i]);
    }

    for (int i = 0; i < n; i++) {
        double div = A[i][i];
        for (int j = 0; j <= n; j++) {
            A[i][j] /= div;
        }

        for (int j = 0; j < n; j++) {
            if (j != i) {
                double mult = A[j][i];
                for (int k = 0; k <= n; k++) {
                    A[j][k] -= mult * A[i][k];
                }
            }
        }
    }

    std::vector<double> x(n);
    for (int i = 0; i < n; i++) {
        x[i] = A[i][n];
    }
    return x;
}

std::vector<double> weightedPolynomialFit(const std::vector<std::tuple<double, double, double>> &points, int k) {
    int n = points.size();
    std::vector<std::vector<double>> A(k + 1, std::vector<double>(k + 1, 0.0));
    std::vector<double> b(k + 1, 0.0);

    for (int i = 0; i <= k; ++i) {
        for (int j = 0; j <= k; ++j) {
            for (const auto &point: points) {
                double x = std::get<0>(point), w = std::get<2>(point);
                A[i][j] += w * std::pow(x, i + j);
            }
        }
        for (const auto &point: points) {
            double x = std::get<0>(point), y = std::get<1>(point), w = std::get<2>(point);
            b[i] += w * std::pow(x, i) * y;
        }
    }

    return gaussJordan(A, b);
}

//Взвешеная полиномиальная аппроксимация
void slidingWindowApproximation(std::vector<std::pair<double, double>> &points, int k = 4, int windowWidth = 17) {

    if (points.size() < windowWidth) {
        std::cerr << "Imposible to approximate, not enough data" << std::endl;
        points.clear();
        return;
    }

    std::vector<std::pair<double, double>> result;
    int halfWindow = windowWidth / 2;

    for (int center = halfWindow; center < points.size() - halfWindow; ++center) {
        int start = center - halfWindow;
        int end = center + halfWindow;

        std::vector<std::tuple<double, double, double>> windowPoints;
        for (int i = start; i <= end; ++i) {
            double weight = exp(-std::pow(4.0 * (i - center), 2) / std::pow(windowWidth - 1, 2));
            windowPoints.push_back(std::make_tuple(points[i].first - points[center].first, points[i].second, weight));
        }

        std::vector<double> coeffs = weightedPolynomialFit(windowPoints, k);
        result.push_back({points[center].first, coeffs[0]});
    }

    points = std::move(result);
}

//диференцирование
void differentiateValuesInPlace(std::vector<std::pair<double, double>> &points, int shift = 2) {
    if (points.size() < shift + 1) {
        // Недостаточно точек для вычисления разности
        std::cerr << "Imposible to differentiate, not enough data" << std::endl;
        points.clear();
        return;
    }

    for (size_t i = 0; i < points.size() - 2; ++i) {
        double x1 = points[i].first;
        double y1 = points[i].second;
        double x2 = points[i + 2].first;
        double y2 = points[i + 2].second;

        // Разность значений y
        double deltaY = y2 - y1;
        double deltaX = x2 - x1;
        // Среднее значение x
        double meanX = (x1 + x2) / 2.0;

        // Обновляем текущую точку в массиве
        points[i] = {meanX, deltaY / deltaX};
    }

    // Удаляем последнюю точку, так как она не участвует в вычислении разности
    points.pop_back();
    points.pop_back();

    auto maxY = std::max_element(points.begin(), points.end(),
                                 [](const std::pair<double, double> &a, const std::pair<double, double> &b) {
                                     return std::abs(a.second) < std::abs(b.second);
                                 })->second;

    if (maxY > 20) {
        std::cerr
                << "derivative of values is to large, results may not be correct. Increase the bin size or try other startup parameters"
                << std::endl;
    }
}

void trimValues(std::vector<std::pair<double, double>> &points, int window_size = 20) {

    //Обрезает пиксели, растояния до линии от которых больше чем window_size
    //В случае, если значение не положительно, то работает адаптивные режим.

    if (window_size > 0) {
        auto newEnd = std::remove_if(points.begin(), points.end(),
                                     [window_size](const std::pair<double, double> &point) {
                                         return std::abs(point.first) > window_size;
                                     });

        points.erase(newEnd, points.end());
        return;
    }
}

void normalizeY(std::vector<std::pair<double, double>> &points) {
    if (points.empty()) {
        return; // Проверка на пустой вектор
    }

    // Находим максимальное значение y
    double maxY = points[0].second;
    for (const auto &point: points) {
        if (point.second > maxY) {
            maxY = point.second;
        }
    }

    // Нормализуем значения y, делая максимальное значение равным 1
    for (auto &point: points) {
        point.second /= maxY;
    }
}

//окно Ханна
void applyHannWindow(std::vector<std::pair<double, double>> &points, double window_size=15) {
    // Шаг 1: Удаление точек вне заданной ширины окна
    auto newEnd = std::remove_if(points.begin(), points.end(), [window_size](const std::pair<double, double> &point) {
        return std::abs(point.first) > window_size;
    });

    points.erase(newEnd, points.end());

    // Шаг 2: Применение окна Ханна к оставшимся точкам
    for (auto &point: points) {
        double x = point.first;
        double w = 0.5 * std::cos(M_PI * x / (window_size * 2));
        point.second *= w; // Применяем окно к значению y
    }
}

//Преобразование фурье
void discreteFourierTransform(std::vector<std::pair<double, double>> &points) {
    int N = points.size();
    std::vector<std::pair<double, double>> result; // Результат ДПФ будет здесь (комплексные числа)

    for (double k = 0; k < 0.5; k += (1.0 / N)) { // Для каждой частоты k (cycles/pixel)
        std::complex<double> sum(0.0, 0.0);
        for (int n = 0; n < N; ++n) { // Для каждой точки во времени n
            double angle = 2;
            angle *= M_PI;
            angle *= k;
            angle *= points[n].first;
            sum += points[n].second * std::exp(std::complex<double>(0, -angle));
        }
        result.push_back(std::make_pair(k, std::abs(sum)));
    }
    points = std::move(result);
}

double findMTFvalue(const std::vector<std::pair<double, double>>& points, double yValue) {
    for (const auto& point : points) {
        if (point.second < yValue) {
            return point.first; // Возвращаем x точки, где y меньше заданного значения
        }
    }
    return std::numeric_limits<double>::infinity();
}

void adjustPointsBasedOnSign(std::vector<std::pair<double, double>>& points) {
    if (points.empty()) {
        return; // Возвращаемся, если массив точек пуст
    }

    // Вычисляем математическое ожидание по y
    double meanY = std::accumulate(points.begin(), points.end(), 0.0,
                                   [](double sum, const std::pair<double, double>& point) {
                                       return sum + point.second;
                                   }) / points.size();

    // Если матожидание отрицательное, меняем y на противоположное
    if (meanY < 0) {
        for (auto& point : points) {
            point.second = -point.second;
        }
    }
}

double detectPeak(std::vector<std::pair<double, double>> points) {
    adjustPointsBasedOnSign(points);
    if (points.size() < 3) {
        return 0.0;
    }

    // Находим три точки с максимальным значением y
    std::vector<std::tuple<double, double, double>> topPoints;
    for (const auto& point : points) {
        topPoints.emplace_back(point.first, point.second, 1.0); // Веса равны 1
    }
    std::sort(topPoints.begin(), topPoints.end(), [](const auto& a, const auto& b) {
        return std::get<1>(a) > std::get<1>(b);
    });
    topPoints.resize(3); // Оставляем только три точки с наибольшим y

    // Вычисляем среднее значение x
    double meanX = std::accumulate(topPoints.begin(), topPoints.end(), 0.0,
                                   [](double sum, const auto& point) {
                                       return sum + std::get<0>(point);
                                   }) / topPoints.size();

    // Вычитаем среднее значение x из каждой точки
    for (auto& point : topPoints) {
        std::get<0>(point) -= meanX;
    }

    // Выполняем полиномиальную аппроксимацию 2-ого порядка
    std::vector<double> coefficients = weightedPolynomialFit(topPoints, 2);

    // Убеждаемся, что парабола ориентирована ветвями вниз
    if (coefficients[2] > 0) {
        return 0.0;
    }

    // Вычисляем координату x вершины параболы и корректируем её
    double vertexX = -coefficients[1] / (2 * coefficients[2]) + meanX;

    return vertexX;
}

double calculateDeviationFactor (std::string file_path, std::pair<double, double> first_cross_points, std::pair<double, double> second_cross_points) {
    auto points = calculateDistAndBrightness(file_path, std::vector<std::pair<double, double>>{first_cross_points, second_cross_points});
    binarize(points, 1);
    trimValues(points, 100);
    slidingWindowApproximation(points, 3, int(points.size()/4)*2-1);
    differentiateValuesInPlace(points);
    slidingWindowApproximation(points, 3, int(points.size()/4)*2-1);
    return detectPeak(points);
};

void movePoints (std::string file_path, std::vector<pair<double, double>> &cross_points, double factor = 1.0) {
    double moving_factor = calculateDeviationFactor(file_path, cross_points[0], cross_points[1])*factor;
    double a = Line(cross_points[0], cross_points[1]).getA();
    double b = Line(cross_points[0], cross_points[1]).getB();
    double ln = sqrt(a * a + b * b);
    a = a / ln;
    b = b / ln;
    cross_points[0].first += a * moving_factor;
    cross_points[1].first += a * moving_factor;
    cross_points[0].second += b * moving_factor;
    cross_points[1].second += b * moving_factor;
}

void rotatePoints (std::string file_path, std::vector<pair<double, double>> &cross_points, double factor = 1.0) {

    std::pair<double, double> middle_point = std::make_pair(
            (cross_points[0].first + cross_points[1].first) / 2,
            (cross_points[0].second + cross_points[1].second) / 2);

    double DeviationFactorL = calculateDeviationFactor(file_path, cross_points[0], middle_point);
    double DeviationFactorR = calculateDeviationFactor(file_path, middle_point, cross_points[1]);

    double rotation_factor = (DeviationFactorL - DeviationFactorR) * factor;



        double prev_len = sqrt((cross_points[0].first - cross_points[1].first) * (cross_points[0].first - cross_points[1].first) +
                           (cross_points[0].second - cross_points[1].second) * (cross_points[0].second - cross_points[1].second));

    double a = Line(cross_points[0], cross_points[1]).getA();
    double b = Line(cross_points[0], cross_points[1]).getB();
    double ln = sqrt(a * a + b * b);
    a = a / ln;
    b = b / ln;
    cross_points[0].first += a * rotation_factor;
    cross_points[1].first -= a * rotation_factor;
    cross_points[0].second += b * rotation_factor;
    cross_points[1].second -= b * rotation_factor;

    double paralel_x = cross_points[1].first - cross_points[0].first;
    double paralel_y = cross_points[1].second - cross_points[0].second;

    double real_len = sqrt(paralel_y*paralel_y+paralel_x*paralel_x);
    double to_remove = (real_len-prev_len)/2;

    double tmp_ln  = sqrt(paralel_x*paralel_x+paralel_y*paralel_y);
    paralel_x = paralel_x/tmp_ln;
    paralel_y = paralel_y/tmp_ln;

    cross_points[0].first += paralel_x * to_remove;
    cross_points[0].second += paralel_y * to_remove;

    cross_points[1].first -= paralel_x * to_remove;
    cross_points[1].second -= paralel_y * to_remove;
}

//функция для уточнения координат
void pinPoint(std::string file_path, std::vector<pair<double, double>> &cross_points){
    movePoints(file_path, cross_points);
    rotatePoints(file_path, cross_points, 0.9);
    rotatePoints(file_path, cross_points, 1);
    movePoints(file_path, cross_points);
}

//Эти функции для удобного сохранения результатов обработки на всех этапах. Используется лишь в случае, если осуществляется логирование.

//Создаёт или обновляет папку для логирования
std::string PrepareLogDir(const std::string &filePath) {
    fs::path pathToFile(filePath);
    fs::path dir = pathToFile.parent_path();
    fs::path processingLogDir = dir / "processing_log";

    if (fs::exists(processingLogDir)) {
        fs::path oldDir = processingLogDir / "old";
        if (!fs::exists(oldDir)) {
            fs::create_directory(oldDir);
        }

        bool hasNonOldContents = false;
        for (const auto &entry: fs::directory_iterator(processingLogDir)) {
            if (entry.path().filename() != "old") {
                hasNonOldContents = true;
                break;
            }
        }

        if (hasNonOldContents) {
            int folderNumber = 0;
            fs::path newFolder;
            do {
                newFolder = oldDir / std::to_string(folderNumber++);
            } while (fs::exists(newFolder));

            fs::create_directory(newFolder);

            for (const auto &entry: fs::directory_iterator(processingLogDir)) {
                if (entry.path().filename() != "old") {
                    fs::path destination = newFolder / entry.path().filename();
                    fs::rename(entry.path(), destination);
                }
            }
        }
    } else {
        fs::create_directory(processingLogDir);
        fs::path oldDir = processingLogDir / "old";
        fs::create_directory(oldDir);

        // Копирование файла, указанного в filePath, в папку processingLogDir
    }
    fs::copy(filePath, processingLogDir / pathToFile.filename(), fs::copy_options::overwrite_existing);
    return processingLogDir.string();
}

//Сохранят точки в удобном формате
void savePointsToFile(const std::string &directoryPath,
                      const std::string &fileName,
                      const std::vector<std::pair<double, double>> &points, bool is_log = true) {

    if (!is_log) return;
    if (points.empty()) return;

    fs::path filePath = fs::path(directoryPath) / (fileName + ".txt");
    std::ofstream outFile(filePath);

    if (!outFile.is_open()) {
        throw std::runtime_error("Failed to open file: " + filePath.string());
    }

    for (const auto &pair: points) {
        outFile << "(" << pair.first << "," << pair.second << ")\n";
    }

    outFile.close();
}

void replaceCharacters(std::string &str, char findChar, char replaceChar) {
    std::replace(str.begin(), str.end(), findChar, replaceChar);
}

//Сохраняет графики
void savePlot(const std::string &directoryPath, std::string fileName,
              const std::string &xlabel, const std::string &ylabel,
              const std::vector<std::pair<double, double>> &points, bool is_log = true) {

    if (!is_log) return;
    if (points.empty()) return;

    std::vector<double> x, y;
    for (const auto &pair: points) {
        x.push_back(pair.first);
        y.push_back(pair.second);
    }

    plt::figure_size(1200, 780);
    plt::plot(x, y, "ro");

    plt::xlabel(xlabel.c_str());
    plt::ylabel(ylabel.c_str());

    std::string title = fileName;
    replaceCharacters(title, '_', ' ');
    plt::title(title.c_str());

    replaceCharacters(fileName, ' ', '_');
    std::string filePath = directoryPath + "/" + fileName + ".png";
    plt::save(filePath);
}

//Сохраняет картинку с изображением крестов, установленых пользователем
void updateImageWithCrosses(const string &imagePath, const vector<pair<double, double>> &crossCoords,
                            const string &directoryPath, const string &newFileName = "crosses") {
    // Загружаем изображение из указанного пути
    Mat img = imread(imagePath);
    if (img.empty()) {
        cout << "Ошибка: не удалось загрузить изображение." << endl;
        return;
    }

    Mat imgCopy = img.clone(); // Создаем копию исходного изображения
    for (auto &coord: crossCoords) {
        int x = int(coord.first);
        int y = int(coord.second);
        int size = 15; // Размер крестика
        Scalar color = Scalar(0, 0, 255); // Цвет крестика (красный)
        int thickness = 3; // Толщина линий крестика

        // Рисуем вертикальную линию крестика
        line(imgCopy, Point(x - size, y), Point(x + size, y), color, thickness);
        // Рисуем горизонтальную линию крестика
        line(imgCopy, Point(x, y - size), Point(x, y + size), color, thickness);
    }

    // Формируем путь сохранения изображения
    string savePath = directoryPath + "/" + newFileName + ".png"; // Добавляем расширение файла
    // Сохраняем обновленное изображение в указанную директорию под новым именем
    if (!imwrite(savePath, imgCopy)) {
        cout << "Ошибка при сохранении изображения." << endl;
    }
}

void process_data(std::string file_path, vector<pair<double, double>> crossCoords) {

    std::string logDir = "";
    if (LOG_PLOTS || LOG_DATA) logDir = PrepareLogDir(file_path);

    updateImageWithCrosses(file_path, crossCoords, logDir, "crosses");

    auto points = calculateDistAndBrightness(file_path, crossCoords);
    savePointsToFile(logDir, "raw ESF", points, LOG_DATA);
    savePlot(logDir, "raw ESF", "distance", "brightness", points, LOG_PLOTS);

    // Применяем бинаризацию и сохраняем результаты
    binarize(points, 0.2);
    savePointsToFile(logDir, "binarized_ESF", points, LOG_DATA);
    savePlot(logDir, "binarized_ESF", "distance", "brightness", points, LOG_PLOTS);

    // Обрезаем значения и сохраняем результаты
    trimValues(points, 40);
    savePointsToFile(logDir, "trimmed_ESF", points, LOG_DATA);
    savePlot(logDir, "trimmed_ESF", "distance", "brightness", points, LOG_PLOTS);

    // Применяем аппроксимацию скользящим окном и сохраняем результаты
    slidingWindowApproximation(points);
    savePointsToFile(logDir, "approximated_ESF", points, LOG_DATA);
    savePlot(logDir, "approximated_ESF", "distance", "brightness", points, LOG_PLOTS);


    // Дифференцируем значения и сохраняем результаты
    differentiateValuesInPlace(points);
    adjustPointsBasedOnSign(points);
    savePointsToFile(logDir, "LSF", points, LOG_DATA);
    savePlot(logDir, "LSF", "distance", "d(brightness)/d(distance)", points, LOG_PLOTS);


    // Сглаживаем значения с помощью огна хана
    applyHannWindow(points);
    savePointsToFile(logDir, "Smoothed LSF", points, LOG_DATA);
    savePlot(logDir, "Smoothed LSF", "distance", "d(brightness)/d(distance)", points, LOG_PLOTS);


    // Применяем дискретное преобразование Фурье, нормализуем Y и сохраняем результаты
    discreteFourierTransform(points);
    normalizeY(points);
    savePointsToFile(logDir, "MTF", points, LOG_DATA);
    savePlot(logDir, "MTF", "Spatial Frequency [cycles/pixel]", "TTF", points, LOG_PLOTS);

    if (LOG_PLOTS || LOG_DATA) std::cout << "All data was saved to\t" << logDir << std::endl;

    double mtf05 = findMTFvalue(points, 0.5);
    double mtf01 = findMTFvalue(points, 0.1);
    std::cout << "mtf_0.5:\t" << mtf05 << "\tmtf_0.1:\t" << mtf01 << std::endl;
}