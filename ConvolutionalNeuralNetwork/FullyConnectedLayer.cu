#include <iostream>

#include "Hub.h"
#include "Random.h"
#include "CustomException.h"
#include "FullyConnectedLayer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 256
#define DOUBLE_BLOCK_SIZE 32
#define LearningRate 0.0005f

texture<float, 1, cudaReadModeElementType> InputsRef;
texture<float, 1, cudaReadModeElementType> GradientsRef;

__global__ void cuda_find_max(float* A, float* max, const int size)
{
	float max_val = _I32_MIN;
	for (int i = 0; i < size; i++)
	{
		max_val = __max(A[i], max_val);
	}
	max[0] = max_val;
}

__global__ void cuda_exp_vector_generate(float* A, float* B, float* max, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		B[idx] = __expf(A[idx] - max[0]);
	}
}

__global__ void cuda_softmax(float* A, float* max, const float log_val, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		A[idx] = __expf(A[idx] - max[0] - log_val);
	}
}

__global__ void cuda_sigmoid(float* outputs, const int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
	{
		outputs[idx] = 1 / (1 + __expf(-1 * outputs[idx]));
	}
}

__global__ void cuda_set_gradients(float* gradients, float* outputs, const int num)
{
	int idx = threadIdx.x;
	gradients[idx] = outputs[idx];
	if (idx == num)
		gradients[idx] = outputs[idx] - 1;
}

__global__ void cuda_gr_to_der_mult(float* gradients, const int in_size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < in_size)
	{
		float inp_elem = tex1Dfetch(InputsRef, idx);
		gradients[idx] *= (inp_elem * (1 - inp_elem));
	}
}

__global__ void cuda_correct_weights(float* weights, const int inp_count, const int out_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < inp_count && idy < out_count)
	{
		weights[idy * inp_count + idx] -= LearningRate * tex1Dfetch(InputsRef, idx) * tex1Dfetch(GradientsRef, idy);
	}
}

__global__ void cuda_correct_biases(float* biases, const int out_count)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < out_count)
	{
		biases[idx] -= LearningRate * tex1Dfetch(GradientsRef, idx);
	}
}

FullyConnectedLayer::FullyConnectedLayer(int in_size, int out_size, Hub& params_storage, ActivationType type) : inputs_device(nullptr) {
	
	network_error = 0.0f;
	this->in_size = in_size;
	this->out_size = out_size;
	this->type = type;
	cudaMalloc((void**)&gradients_device, out_size * sizeof(float));
	cudaMalloc((void**)&outputs_device, out_size * sizeof(float));
	cublasCreate(&handle);

	if (params_storage.get_status() == Status::Training)
	{
		weights_device = set_normal_random(in_size * out_size, 1, weights_pitch);
		biases_device = set_repeatable_values(out_size, 0.01f);
	}
	else
	{
		weights_device = params_storage.get_params(in_size * out_size);
		biases_device = params_storage.get_params(out_size);
	}
	
}

float* FullyConnectedLayer::forward(float* prev_layer_data) {

	inputs_device = prev_layer_data;
	m_v_multiplication(weights_device, inputs_device, outputs_device, handle);
	add_biases(handle);
	activate(handle);
	return outputs_device;
}

void FullyConnectedLayer::backward(float* prev_layer_gradients) {
	
	cudaBindTexture(0, InputsRef, inputs_device, in_size * sizeof(float));
	cudaBindTexture(0, GradientsRef, gradients_device, out_size * sizeof(float));

	m_v_multiplication(weights_device, gradients_device, prev_layer_gradients, handle, CUBLAS_OP_N);

	dim3 threadsPerBlock = BLOCK_SIZE;
	dim3 blocksPerGrid = in_size / BLOCK_SIZE + (in_size % BLOCK_SIZE == 0 ? 0 : 1);

	cuda_gr_to_der_mult << <blocksPerGrid, threadsPerBlock >> > (prev_layer_gradients, in_size);
	cudaDeviceSynchronize();
	cudacall(cudaGetLastError());

	correct();

	cudaUnbindTexture(InputsRef);
	cudaUnbindTexture(GradientsRef);
}

void FullyConnectedLayer::correct() {
	
	dim3 threadsPerBlock = dim3(DOUBLE_BLOCK_SIZE, DOUBLE_BLOCK_SIZE);
	dim3 blocksPerGrid = dim3(in_size / DOUBLE_BLOCK_SIZE + (in_size % DOUBLE_BLOCK_SIZE == 0 ? 0 : 1),
		out_size / DOUBLE_BLOCK_SIZE + (out_size % DOUBLE_BLOCK_SIZE == 0 ? 0 : 1));
	cuda_correct_weights << <blocksPerGrid, threadsPerBlock >> > (weights_device, in_size, out_size);
	cudacall(cudaGetLastError());

	threadsPerBlock = BLOCK_SIZE;
	blocksPerGrid = out_size / BLOCK_SIZE + (out_size % BLOCK_SIZE == 0 ? 0 : 1);
	cuda_correct_biases << <blocksPerGrid, threadsPerBlock >> > (biases_device, out_size);
	cudacall(cudaGetLastError());
}

void FullyConnectedLayer::m_v_multiplication(float* matrix, float* vector, float* result_vector, cublasHandle_t& handle, cublasOperation_t trans) {
	
	float alpha = 1.0f, beta = 0.0f;
	cublascall(cublasSgemv(handle, trans, in_size, out_size, &alpha, matrix, in_size, vector, 1, &beta, result_vector, 1));
}

void FullyConnectedLayer::add_biases(cublasHandle_t& handle) {
	
	float alpha = 1.0f;
	cublascall(cublasSaxpy(handle, out_size, &alpha, biases_device, 1, outputs_device, 1));
}

void FullyConnectedLayer::activate(cublasHandle_t& handle) {
	
	if (type == ActivationType::Softmax)
		activate_softmax(handle);
	else
		activate_sigmoid();
}

void FullyConnectedLayer::activate_sigmoid() {
	
	dim3 threadsPerBlock = BLOCK_SIZE;
	dim3 blocksPerGrid = out_size / BLOCK_SIZE + (out_size % BLOCK_SIZE == 0 ? 0 : 1);
	cuda_sigmoid << <blocksPerGrid, threadsPerBlock >> > (outputs_device, out_size);
	cudacall(cudaGetLastError());
}

void FullyConnectedLayer::activate_softmax(cublasHandle_t& handle) {
	
	float sum = 0.0f;
	float* max_device;
	float* helper_vector_device;
	dim3 threadsPerBlock = BLOCK_SIZE;
	dim3 blocksPerGrid = out_size / BLOCK_SIZE + (out_size % BLOCK_SIZE == 0 ? 0 : 1);
	
	cudaMalloc((void**)&max_device, sizeof(float));
	cudaMalloc((void**)&helper_vector_device, out_size * sizeof(float));
	cuda_find_max << <1, 1>> > (outputs_device, max_device, out_size);
	cudacall(cudaGetLastError());

	cuda_exp_vector_generate << <blocksPerGrid, threadsPerBlock >> > (outputs_device, helper_vector_device, max_device, out_size);
	cudacall(cudaGetLastError());

	cublascall(cublasSasum(handle, out_size, helper_vector_device, 1, &sum));

	sum = log(sum);
	cuda_softmax << <blocksPerGrid, threadsPerBlock >> > (outputs_device, max_device, sum, out_size);
	cudacall(cudaGetLastError());
	
	cudaFree(helper_vector_device);
	cudaFree(max_device);
}

void FullyConnectedLayer::set_gradients(int correct_result) {

	cuda_set_gradients << <1, 10 >> > (gradients_device, outputs_device, correct_result);
	cudacall(cudaGetLastError());
}

float* FullyConnectedLayer::get_gradients() {
	
	return gradients_device;
}

int FullyConnectedLayer::get_result() {
	
	int max_index;
	cublascall(cublasIsamax(handle, out_size, outputs_device, 1, &max_index));
	return max_index - 1;
}

void FullyConnectedLayer::calc_error(int correct_result) {

	float local_error;
	cudaMemcpy(&local_error, outputs_device + correct_result, sizeof(float), cudaMemcpyDeviceToHost);
	network_error -= log(local_error);
}

float FullyConnectedLayer::get_common_error(const int set_size) {

	return network_error / set_size;
}

void FullyConnectedLayer::save_params(Hub& params_storage) {

	params_storage.set_params(weights_device, in_size * out_size);
	params_storage.set_params(biases_device, out_size);
}

void FullyConnectedLayer::freeInputs() {

	cudaFree(inputs_device);
	cudacall(cudaGetLastError());
}

void FullyConnectedLayer::freeMemory() {
	
	cudaFree(gradients_device);
	cudaFree(weights_device);
	cudaFree(biases_device);
	cudaFree(inputs_device);
	cublasDestroy(handle);
}