#pragma once

#include "Tensor.h"

class ConvolutionalLayer {
private:
	Tensor inputs_device;
	Tensor filters_device;
	Tensor outputs_devices;
	float* biases_device;
	void convolve();
	void activate();

public:
	Tensor filters_gr_device;
	float* biases_gr_device;

	ConvolutionalLayer(const int filters_size, const int filters_count, const int gradients_size, const int gradients_depth);
	Tensor& forward(Tensor& input_matrixes);
	void backward(Tensor& prev_gradient_matrixes);
	void correct();
	void freeMemory();
};