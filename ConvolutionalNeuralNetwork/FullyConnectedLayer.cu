#include <malloc.h>
#include <iostream>

#include "Random.h"
#include "FullyConnectedLayer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 32

texture<float, 1, cudaReadModeElementType> InputsRef;

__global__ void cuda_multiply_matrixes(float* matrix, const int cols, const int rows)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < cols && y < rows)
	{
		matrix[y * cols + x] *= tex1Dfetch(InputsRef, x);
	}
}

__global__ void cuda_sum_particles(float* A, float* B, const int size, const int cols, const int rows, const int iterations)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int ydx = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < iterations && ydx < rows)
	{
		int offset = ydx * cols;
		int start = idx * BLOCK_SIZE;
		int end = __min((idx + 1) * BLOCK_SIZE, size);
		A[offset + idx] = 0;
		for (int j = start; j < end; j++)
			A[offset + idx] = __fadd_rn(A[offset + idx], B[offset + j]);
	}
}

__global__ void cuda_exp_vector_generate(float* A, float* B, float max, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		B[idx] = __expf(A[idx] - max);
	}
}

__global__ void cuda_softmax(float* A, float max, const float log_val, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{

		A[idx] = __expf(A[idx] - max - log_val);
	}
}

float find_max(float* arr, int size);

FullyConnectedLayer::FullyConnectedLayer(int in_size, int out_size) {
	this->in_size = in_size;
	this->out_size = out_size;
	weights_matrix = (float*)malloc(sizeof(float) * in_size * out_size);
	biases_vector = (float*)malloc(sizeof(float) * out_size);
	set_normal_random(weights_matrix, in_size * out_size);
	set_normal_random(biases_vector, out_size);

	//test
	/*for (int i = 0; i < in_size * out_size; i++)
		weights_matrix[i] = i;*/
}

float* FullyConnectedLayer::forward(float* prev_layer_data) {
	float* outputs;
	float* inputs_device;
	float* outputs_device;
	float* weights_device;
	cublasHandle_t handle;
	cublasCreate(&handle);
	rsize_t size = in_size * sizeof(float);
	cudaMalloc((void**)&inputs_device, size);
	cudaMalloc((void**)&weights_device, out_size * size);
	outputs = (float*)malloc(out_size * sizeof(float));

	cudaMemcpy(inputs_device, prev_layer_data, size, cudaMemcpyHostToDevice);
	cudaMemcpy(weights_device, weights_matrix, out_size * size, cudaMemcpyHostToDevice);
	cudaBindTexture(0, InputsRef, inputs_device, InputsRef.channelDesc, size);

	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(in_size / BLOCK_SIZE + (in_size % BLOCK_SIZE == 0 ? 0 : 1), out_size / BLOCK_SIZE + (out_size % BLOCK_SIZE == 0 ? 0 : 1));

	cuda_multiply_matrixes << <blocksPerGrid , threadsPerBlock >> > (weights_device, in_size, out_size);
	outputs_device = sum_particles_host(weights_device, in_size, out_size, BLOCK_SIZE);
	add_biases(outputs_device, handle);
	activate_softmax(outputs_device, outputs, handle);

	cudaUnbindTexture(InputsRef);
	cudaFree(inputs_device);
	cudaFree(outputs_device);
	cudaFree(weights_device);
	cublasDestroy(handle);

	return outputs;
}

void FullyConnectedLayer::add_biases(float* outputs_device, cublasHandle_t& handle) {
	float alpha = 1.0f;
	float* biases_vector_device;
	cublasSetVector(out_size, sizeof(float), biases_vector, 1, biases_vector_device, 1);
	cublasSaxpy(handle, out_size, &alpha, biases_vector_device, 1, outputs_device, 1);
	cudaFree(biases_vector_device);
}

void FullyConnectedLayer::activate_softmax(float* outputs_device, float* outputs, cublasHandle_t& handle) {
	float sum = 0.0f;
	float* helper_vector_device;
	dim3 threadsPerBlock = 256;
	dim3 blocksPerGrid = out_size / 256 + (out_size % 256 == 0 ? 0 : 1);
	
	cudaMalloc((void**)&helper_vector_device, out_size * sizeof(float));
	cudaMemcpy(outputs, outputs_device, out_size * sizeof(float), cudaMemcpyDeviceToHost);
	
	float max_val = find_max(outputs, out_size);
	cuda_exp_vector_generate << <blocksPerGrid, threadsPerBlock >> > (outputs_device, helper_vector_device, max_val, out_size);
	cublasSasum(handle, out_size, helper_vector_device, 1, &sum);
	sum = log(sum);
	cuda_softmax << <blocksPerGrid, threadsPerBlock >> > (outputs_device, max_val, sum, out_size);
	
	cudaMemcpy(outputs, outputs_device, out_size * sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaFree(helper_vector_device);
}

float* FullyConnectedLayer::sum_particles_host(float* d_A_odd, int in_size, int out_size, int rows) {
	bool isOdd = true;
	int arr_length = in_size;
	float* d_A_even;
	cudaMalloc((void**)&d_A_even, out_size * in_size * sizeof(float));
	dim3 threadsPerBlock = dim3(BLOCK_SIZE, rows);
	int iterations = arr_length / BLOCK_SIZE + (arr_length % BLOCK_SIZE != 0 ? 1 : 0);
	while (arr_length != 1)
	{
		dim3 blocksPerGrid = dim3(iterations / BLOCK_SIZE + (iterations % BLOCK_SIZE != 0 ? 1 : 0), out_size / BLOCK_SIZE + (out_size % BLOCK_SIZE == 0 ? 0 : 1));
		if (isOdd)
			cuda_sum_particles << <blocksPerGrid, threadsPerBlock >> > (d_A_even, d_A_odd, arr_length, in_size, out_size, iterations);
		else
			cuda_sum_particles << <blocksPerGrid, threadsPerBlock >> > (d_A_odd, d_A_even, arr_length, in_size, out_size, iterations);
		cudaDeviceSynchronize();

		arr_length = iterations;
		iterations = arr_length / BLOCK_SIZE + (arr_length % BLOCK_SIZE != 0 ? 1 : 0);
		isOdd = !isOdd;
	}
	float* result;
	cudaMalloc((void**)&result, sizeof(float) * out_size);
	cudaMemcpy2D(result, sizeof(float), isOdd ? d_A_odd : d_A_even, in_size * sizeof(float), sizeof(float), out_size, cudaMemcpyDeviceToDevice);
	cudaFree(d_A_even);

	return result;
}

float find_max(float* arr, int size)
{
	float max = _I32_MIN;
	for (int i = 0; i < size; i++)
	{
		max = __max(arr[i], max);
	}
	return max;
}