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

	void add_biases(cublasHandle_t& handle);
	void activate_softmax(cublasHandle_t& handle);
	void m_v_multiplication(float* matrix, float* vector, float* result_vector, cublasHandle_t& handle);

public:
	FullyConnectedLayer(int in_size, int out_size);
	float* forward(float* prev_layer_data);
	void freeMemory();
};