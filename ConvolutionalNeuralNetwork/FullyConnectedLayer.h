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
	float* weights_device;
	float* w_velocity_device;
	float* accelerated_w_device;
	float* biases_device;
	float* b_velocity_device;
	float network_error;

	float* sum;
	float* max_device;
	cublasHandle_t handle;

	void correct();
	void add_biases(cublasHandle_t& handle);
	void activate(cublasHandle_t& handle);
	void activate_sigmoid();
	void activate_softmax(cublasHandle_t& handle);
	void give_speed(float* weights_device, float* w_velocity_device, float* accelerated_w_device);
	void m_v_multiplication(float* matrix, float* vector, float* result_vector, cublasHandle_t& handle, cublasOperation_t trans = CUBLAS_OP_T);

public:
	float* inputs_device;
	float* outputs_device;
	float* gradients_device;

	FullyConnectedLayer(int in_size, int out_size, Hub& params_storage, ActivationType type = ActivationType::Sigmoid);
	void set_gradients(int correct_result);
	float* forward(float* prev_layer_data);
	void backward(float* prev_layer_gradients);
	int get_result();
	void calc_error(int correct_result);
	float get_common_error(const int set_size);
	void save_params(Hub& params_storage);
	void freeMemory();
};