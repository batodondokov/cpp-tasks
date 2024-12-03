#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h> 
#include <thrust/count.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <curand.h>

#include <chrono>

void usage(const char* filename)
{
	printf("Алгоритм поиска площади криволинейной трапеции используя метод Монте-Карло.\n");
	printf("Usage: %s <n>\n", filename);
}

// Функтор для нахождения значения функции
struct fx_functor
{
	__host__ __device__
    float operator()(float x) const
	{
		return  x * x / (x + 1) + 1 / x;
	}

};

// Функтор для масштабирования значений полученных в результате генерации случайных чисел.
struct scale
{
    float a, b;

    scale(float a, float b) : a(a), b(b) {}

    __device__ 
	float operator()(float x) const {
        return a + x * (b - a);
    }
};

// Функтор для сравнения двух массивов
struct count_under_curve {
    __host__ __device__
    bool operator()(const thrust::tuple<float, float>& t) const 
	{
        return thrust::get<0>(t) >= thrust::get<1>(t);
    }
};

int main(int argc, char* argv[])
{
	// Начинаю отсчет времени
	auto start = std::chrono::high_resolution_clock::now();

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
	cudaSetDevice(2);


    printf("Input data:\n");
    printf("Количество точек: %d\n", n);
    printf("\n");

	// Задаю начальные значения для дальнейших вычислений
	float a = 1.0f;
	float b = 2.0f;
	float max_fx_value = 0.0f;
	int under_curve = 0;

	// Генерирую случайные массивы X, Y на GPU
    curandGenerator_t gen;

    float *dev_X_rand , *dev_Y_rand;

    cudaMalloc (( void **)&dev_X_rand , n* sizeof( float ));
    cudaMalloc (( void **)&dev_Y_rand , n* sizeof( float ));

    curandCreateGenerator(&gen , CURAND_RNG_PSEUDO_DEFAULT );
    curandSetPseudoRandomGeneratorSeed(gen , 12345ULL);

    curandGenerateUniform(gen , dev_X_rand , n);
    curandGenerateUniform(gen , dev_Y_rand , n);

	// Переношу массивы случайных значений в thrust массивы на GPU
    thrust::device_vector<float> dev_X_thrust(dev_X_rand, dev_X_rand + n);
	thrust::device_vector<float> dev_Y_thrust(dev_Y_rand, dev_Y_rand + n);

	// Нахожу максимальное значние функции на отрезке
	thrust::device_vector<float> dev_X_sequence(1001), d_Y_sequence(1001);
	thrust::sequence(dev_X_sequence.begin(), dev_X_sequence.end(), a, 0.001f);
	thrust::transform(dev_X_sequence.begin(), dev_X_sequence.end(), d_Y_sequence.begin(), fx_functor());
	max_fx_value = *(thrust::max_element(d_Y_sequence.begin(), d_Y_sequence.end()));

	// Нахожу площадь достроенного прямоугольника
	float rect_area = max_fx_value * (b - a);

	// Масштабирую массивы случайных значений до нужного дипазона
    thrust::device_vector<float> dev_X_thrust_scaled(n), dev_Y_thrust_scaled(n);;
	thrust::transform(dev_X_thrust.begin(), dev_X_thrust.end(), dev_X_thrust_scaled.begin(), scale(a, b));
	thrust::transform(dev_Y_thrust.begin(), dev_Y_thrust.end(), dev_Y_thrust_scaled.begin(), scale(0, max_fx_value));

	// Нахожу значения функции для случайных X
	thrust::device_vector<float> d_fx(n);
	thrust::transform(dev_X_thrust_scaled.begin(), dev_X_thrust_scaled.end(), d_fx.begin(), fx_functor());

	// Считаю сколько точек находятся под кривой
	under_curve = thrust::count_if(
		thrust::make_zip_iterator(thrust::make_tuple(d_fx.begin(), dev_Y_thrust_scaled.begin())), 
		thrust::make_zip_iterator(thrust::make_tuple(d_fx.end(), dev_Y_thrust_scaled.end())), 
		count_under_curve());	

	// Нахожу площаль трапеции
	float s = rect_area * (under_curve) / n;

	// Останавливаю отсчет времени
	auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    printf("Output data:\n");
	printf("S = %f, time = %f\n", s, duration.count());	

    curandDestroyGenerator(gen); 
	cudaFree( dev_X_rand ); 
	cudaFree( dev_Y_rand ); 

	return 0;
}