#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Random.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CustomException.h"

#define BLOCK_SIZE 256
#define DoublePi 6.28318f

unsigned int seed = time(NULL);

__global__ void generate_normal_random_vector(float* arr, const int arr_size, const float sigma, const unsigned int seed)
{
	curandState_t state;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, id, 0, &state);
	if (2 * id < arr_size)
	{
		const float t1 = sigma * __fsqrt_rn(-2 * __logf(curand_uniform(&state)));
		const float t2 = DoublePi * curand_uniform(&state);

		arr[2 * id] = t1 * __cosf(t2);
		arr[2 * id + 1] = t1 * __sinf(t2);
	}

}

__global__ void set_repeatable_values(float* arr, const int arr_size, const float custom_val)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < arr_size)
		arr[id] = custom_val;
}

float* set_normal_random(const int arr_size, const int depth, size_t& pitch, const float sigma, bool is2dim) {
	
	int common_size = arr_size * depth;
	int half_arr_size = common_size % 2 == 0 ? common_size / 2 : (common_size + 1) / 2;
	float* arr_device;
	const int GRID_SIZE = half_arr_size / BLOCK_SIZE + (half_arr_size % BLOCK_SIZE != 0 ? 1 : 0);

	cudaMalloc((void**)&arr_device, 2 * half_arr_size * sizeof(float));

	generate_normal_random_vector << <GRID_SIZE, BLOCK_SIZE >> > (arr_device, common_size, sigma, seed);
	cudacall(cudaGetLastError());

	float* result_device;

	if (is2dim)
	{
		
		cudaMallocPitch((void**)&result_device, &pitch, arr_size * sizeof(float), depth);
		cudaMemcpy2D(result_device, pitch, arr_device, arr_size * sizeof(float), arr_size * sizeof(float), depth, cudaMemcpyDeviceToDevice);
	}
	else
	{
		cudaMalloc((void**)&result_device, arr_size * sizeof(float));
		cudaMemcpy(result_device, arr_device, arr_size * sizeof(float), cudaMemcpyDeviceToDevice);
	}

	cudaFree(arr_device);
	return result_device;
}

float* set_repeatable_values(const int arr_size, const int depth, size_t& pitch, const float custom_val, bool is2dim) {

	int common_size = arr_size * depth;
	float* arr_device;
	const int GRID_SIZE = common_size / BLOCK_SIZE + (common_size % BLOCK_SIZE != 0 ? 1 : 0);

	cudaMalloc((void**)&arr_device, common_size * sizeof(float));

	set_repeatable_values << <GRID_SIZE, BLOCK_SIZE >> > (arr_device, common_size, custom_val);
	cudacall(cudaGetLastError());

	if (is2dim)
	{
		float* result_device;
		cudaMallocPitch((void**)&result_device, &pitch, arr_size * sizeof(float), depth);
		cudaMemcpy2D(result_device, pitch, arr_device, arr_size * sizeof(float), arr_size * sizeof(float), depth, cudaMemcpyDeviceToDevice);
		cudaFree(arr_device);
		return result_device;
	}
	else
		return arr_device;
}