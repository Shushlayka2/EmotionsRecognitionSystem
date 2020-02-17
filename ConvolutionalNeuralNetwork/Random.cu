#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Random.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CustomException.h"

#define BLOCK_SIZE 256

unsigned int seed = time(NULL);

__global__ void generate_normal_random_vector(float* arr, const int arr_size, const unsigned int seed)
{
	curandState_t state;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, id, 0, &state);
	if (id < arr_size)
		arr[id] = curand_normal(&state);
}

__global__ void set_repeatable_values(float* arr, const int arr_size, const float custom_val)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < arr_size)
		arr[id] = custom_val;
}

float* set_normal_random(const int arr_size, const int depth, size_t& pitch) {
	
	int common_size = arr_size * depth;
	float* arr_device;
	const int GRID_SIZE = common_size / BLOCK_SIZE + (common_size % BLOCK_SIZE != 0 ? 1 : 0);

	cudaMalloc((void**)&arr_device, common_size * sizeof(float));

	generate_normal_random_vector << <GRID_SIZE, BLOCK_SIZE >> > (arr_device, common_size, seed);
	//cudacall(cudaGetLastError());

	if (depth == 1)
		return arr_device;
	else
	{
		float* arr_2d_device;
		cudaMallocPitch((void**)&arr_2d_device, &pitch, arr_size * sizeof(float), depth);
		cudaMemcpy2D(arr_2d_device, pitch, arr_device, arr_size * sizeof(float), arr_size * sizeof(float), depth, cudaMemcpyDeviceToDevice);
		cudaFree(arr_device);

		return arr_2d_device;
	}
}

float* set_repeatable_values(const int arr_size, const float custom_val) {

	float* arr_device;
	const int GRID_SIZE = arr_size / BLOCK_SIZE + (arr_size % BLOCK_SIZE != 0 ? 1 : 0);

	cudaMalloc((void**)&arr_device, arr_size * sizeof(float));

	set_repeatable_values << <GRID_SIZE, BLOCK_SIZE >> > (arr_device, arr_size, custom_val);
	//cudacall(cudaGetLastError());

	return arr_device;
}