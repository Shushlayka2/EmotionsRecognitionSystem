#pragma once

#include <cublas_v2.h>

enum class ActivationType {
	Softmax,
	Sigmoid
};

class FullyConnectedLayer {
private:

	ActivationType type;
	int in_size;
	int out_size;
	float* inputs_device;
	float* outputs_device;
	float* gradients_device;
	float* weights_device;
	float* biases_device;
	float network_error;

	size_t weights_pitch;
	cublasHandle_t handle;

	void correct();
	void add_biases(cublasHandle_t& handle);
	void activate(cublasHandle_t& handle);
	void activate_sigmoid();
	void activate_softmax(cublasHandle_t& handle);
	void m_v_multiplication(float* matrix, float* vector, float* result_vector, cublasHandle_t& handle, cublasOperation_t trans = CUBLAS_OP_T);

public:
	FullyConnectedLayer(int in_size, int out_size, Hub& params_storage, ActivationType type = ActivationType::Sigmoid);
	void set_gradients(int correct_result);
	float* get_gradients();
	float* forward(float* prev_layer_data);
	void backward(float* prev_layer_gradients);
	int get_result();
	void calc_error(int correct_result);
	float get_common_error(const int set_size);
	void save_params(Hub& params_storage);
	void freeInputs();
	void freeMemory();
};