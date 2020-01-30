#pragma once

#include <cublas_v2.h>

class FullyConnectedLayer {
private:
	int in_size;
	int out_size;
	float* inputs_device;
	float* outputs_device;
	float* gradients_device;
	float* weights_device;
	float* biases_device;

	size_t weights_pitch;
	cublasHandle_t handle;

	void add_biases(cublasHandle_t& handle);
	void activate_softmax(cublasHandle_t& handle);
	void m_v_multiplication(float* matrix, float* vector, float* result_vector, cublasHandle_t& handle, cublasOperation_t trans = CUBLAS_OP_T);

public:
	FullyConnectedLayer(int in_size, int out_size);
	void set_gradients(int correct_result);
	float* get_gradients();
	float* forward(float* prev_layer_data);
	void backward(float* prev_layer_gradients);
	void freeMemory();
};