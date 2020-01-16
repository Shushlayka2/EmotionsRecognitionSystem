#include "Random.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Exception.h"
#include <time.h>
#include <curand.h>
#include<iostream>
#include <curand_kernel.h>

#define DoublePi 6.28318f
#define BLOCK_SIZE 256

__global__ void generate_normal_random_vector(float* arr, const int arr_size, const float mu, const float sigma, const unsigned int seed)
{
	curandState_t state;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed, id, 0, &state);

	//Преобразование Бокса — Мюллера для моделирования нормального распределения
	if (2 * id < arr_size)
	{
		const float t1 = sigma * __fsqrt_rn(-2 * __logf(curand_uniform(&state)));
		const float t2 = DoublePi * curand_uniform(&state);

		arr[2 * id] = t1 * __cosf(t2) + mu;
		arr[2 * id + 1] = t1 * __sinf(t2) + mu;
	}
}

void set_normal_random(float* arr, const int arr_size, const float mu, const float sigma)
{
	int half_arr_size = arr_size % 2 == 0 ? arr_size / 2 : (arr_size + 1) / 2;
	float* arr_device;
	const int GRID_SIZE = half_arr_size / BLOCK_SIZE + (half_arr_size % BLOCK_SIZE != 0 ? 1 : 0);

	try
	{
		cudaMalloc((void**)&arr_device, sizeof(float) * half_arr_size * 2);

		generate_normal_random_vector << <GRID_SIZE, BLOCK_SIZE >> > (arr_device, half_arr_size * 2, mu, sigma, time(NULL));
		cudacall(cudaGetLastError());

		cudaMemcpy(arr, arr_device, sizeof(float) * arr_size, cudaMemcpyDeviceToHost);
	}
	catch (custom_exception & ex)
	{
		ex.destroy();
		printf("Exception appeared! See log file!");
		//TODO: Complete program?
	}
}