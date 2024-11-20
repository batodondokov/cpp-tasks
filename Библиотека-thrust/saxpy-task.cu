#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <cstdio>
#include <cstdlib>

void usage(const char* filename)
{
	printf("Calculating a saxpy transform for two random vectors of the given length.\n");
	printf("Usage: %s <n>\n", filename);
}

// using namespace thrust;
//using namespace thrust::placeholders;

// TODO: Please refer to sorting examples:
// http://code.google.com/p/thrust/
// http://code.google.com/p/thrust/wiki/QuickStartGuide#Transformations

struct saxpy
{
	float a;
	// Constructor:
	saxpy(float a): a(a) {}
	// TODO: Define operator ()
	__host__ __device__
	float operator()(float x, float y)
	{
		return a * x + y;
	}

};

int main(int argc, char* argv[])
{
	const int printable_n = 128;

	if (argc != 2)
	{
		usage(argv[0]);
		return 0;
	}

	int n = atoi(argv[1]);
	if (n <= 0)
	{
		usage(argv[0]);
		return 0;
	}
	cudaSetDevice(2);

	// TODO: Generate 3 vectors on host ( z = a * x + y)
	thrust::host_vector<float> X(n), Y(n), Z(n);
	thrust::generate(X.begin(), X.end(), rand);
	thrust::generate(Y.begin(), Y.end(), rand);

	// Print out the input data if n is small.
	if (n <= printable_n)
	{
		printf("Input data:\n");
		for (int i = 0; i < n; i++)
			printf("%f   %f\n", 1.f*X[i] / RAND_MAX, 1.f*Y[i] / RAND_MAX);
		printf("\n");
	}

	// TODO: Transfer data to the device.
	thrust::device_vector<float> X_d = X;
	thrust::device_vector<float> Y_d = Y;
	thrust::device_vector<float> Z_d = Z;

	float a=2.5f;
	// TODO: Use transform to make an saxpy operation
	// Note: you may use placeholders
	thrust::transform(X_d.begin(), X_d.end(), Y_d.begin(), Z_d.begin(), saxpy(a));

	// TODO: Transfer data back to host.
	thrust::copy(X_d.begin(), X_d.end(), X.begin());
	thrust::copy(Y_d.begin(), Y_d.end(), Y.begin());
	thrust::copy(Z_d.begin(), Z_d.end(), Z.begin());

	// Print out the output data if n is small.
	if (n <= printable_n)
	{
		printf("Output data:\n");
		for (int i = 0; i < n; i++)
			printf("%f * %f + %f = %f\n", a, 1.f*X[i] / RAND_MAX, 1.f*Y[i] / RAND_MAX, Z[i] / RAND_MAX);
		printf("\n");
	}

	return 0;
}

