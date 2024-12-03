#include <iostream>
#include <random>
#include <chrono>
#include <random>
#include <algorithm>

void usage(const char* filename)
{
	printf("Алгоритм поиска площади криволинейной трапеции используя метод Монте-Карло.\n");
	printf("Usage: %s <n>\n", filename);
}

// Фукнция для нахождения значения функции
float fx(float x)
{
	return  x * x / (x + 1) + 1 / x;
};

int main(int argc, char* argv[])
{
	// Начинаю отсчет времени
	auto start = std::chrono::high_resolution_clock::now();

	// Задаю начальные значения для дальнейших вычислений
	float a = 1.0f;
	float b = 2.0f;

	if (argc != 2)
	{
		usage(argv[0]);
		return 0;
	}

	size_t n = atoi(argv[1]);
	if (n <= 0)
	{
		usage(argv[0]);
		return 0;
	}
	// Определяю генератор случайных чисел
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist1(a, b);

	// Нахожу максимальное значние функции на отрезке
	int sequence = 1000;
    float X_sequence[sequence], Y_sequence[sequence];
	for (int i = 0; i < sequence; ++i)
	{
		X_sequence[i] = a + float(i+1)/sequence;
		Y_sequence[i] = fx(X_sequence[i]);
	}
	float max_fx_value = *(std::max_element(Y_sequence, Y_sequence+sequence));
    std::uniform_real_distribution<float> dist2(0, max_fx_value);
	
	// Нахожу площадь достроенного прямоугольника
	float rect_area = max_fx_value * (b - a);

	// В цикле нахожу случайное значние для X. Y и посчитываю кол-во точек под кривой
    float X[n], Y[n];
	int under_curve = 0;

    for (int i = 0; i < n; ++i) 
	{
		X[i] = dist1(gen);
		Y[i] = dist2(gen);
		if (Y[i] <= fx(X[i]))
		{
			under_curve++;
		}
	}

	// Нахожу площаль трапеции
	float S = rect_area * (float(under_curve) / n);

	// Останавливаю отсчет времени
	auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    printf("Output data:\n");
	printf("S = %f, time = %f\n", S, duration.count());	
    
    return 0;
}