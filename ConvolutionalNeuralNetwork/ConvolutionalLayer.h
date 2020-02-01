#pragma once

#include "Tensor.h"

class ConvolutionalLayer {
private:
	Tensor inputs_device;
	Tensor filters_device;
	Tensor outputs_devices;
	float* biases_device;

public:
	Tensor filters_gr_device;
	float* biases_gr_device;

	ConvolutionalLayer(const int filters_size, const int filters_count, const int outputs_size, const int outputs_depth);
	Tensor& forward(Tensor& input_matrixes);
	void backward(Tensor& prev_gradient_matrixes);
	void correct();
	void freeMemory();
};