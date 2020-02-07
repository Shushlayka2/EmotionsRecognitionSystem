#pragma once

#include "Tensor.h"

class PoolingLayer {
private:
	unsigned int filter_size;

public:
	Tensor inputs_device;
	Tensor outputs_devices;
	Tensor gradients_device;

	PoolingLayer(const int filter_size, const int outputs_size, const int outputs_depth);
	Tensor& forward(Tensor& input_matrixes, Tensor& prev_gradient_matrixes);
	void backward(Tensor& prev_gradient_matrixes);
	void freeMemory();
};