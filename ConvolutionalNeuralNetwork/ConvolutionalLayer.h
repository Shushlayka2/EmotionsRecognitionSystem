#pragma once

#include "Tensor.h"

class ConvolutionalLayer {
private:
	Tensor filters_device;
	Tensor f_velocity_device;
	Tensor accelerated_f_device;
	float* biases_device;
	float* b_velocity_device;

	void give_speed(Tensor& filters_device, Tensor& f_velocity_device, Tensor& accelerated_f_device);

public:
	Tensor inputs_device;
	Tensor outputs_devices;
	Tensor gradients_device;

	ConvolutionalLayer(const int filters_size, const int filters_count, const int inputs_depth, const int outputs_size, Hub& params_storage);
	Tensor& forward(Tensor& input_matrixes);
	void backward(Tensor& prev_gradient_matrixes);
	void correct();
	void save_params(Hub& params_storage);
	void freeMemory();
};