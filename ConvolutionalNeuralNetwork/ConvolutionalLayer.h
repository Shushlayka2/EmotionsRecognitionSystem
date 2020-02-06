#pragma once

#include "Tensor.h"

class ConvolutionalLayer {
private:
	Tensor inputs_device;
	Tensor filters_device;
	Tensor outputs_devices;
	Tensor gradients_device;
	float* biases_device;

public:

	ConvolutionalLayer(const int filters_size, const int filters_count, const int outputs_size, const int outputs_depth, Hub& params_storage);
	Tensor& forward(Tensor& input_matrixes);
	void backward(Tensor& prev_gradient_matrixes);
	void correct();
	Tensor& get_gradients();
	void save_params(Hub& params_storage);
	void freeInputs();
	void freeMemory();
};