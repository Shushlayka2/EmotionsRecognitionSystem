#pragma once

#include "Tensor.h"

class ConvolutionalLayer {
private:
	Tensor filters_device;
	float* biases_device;

public:
	Tensor inputs_device;
	Tensor outputs_devices;
	Tensor gradients_device;

	ConvolutionalLayer(const int filters_size, const int filters_count, const int outputs_size, const int outputs_depth, Hub& params_storage);
	Tensor& forward(Tensor& input_matrixes);
	void backward(Tensor& prev_gradient_matrixes);
	void correct();
	void save_params(Hub& params_storage);
	void freeMemory();
};