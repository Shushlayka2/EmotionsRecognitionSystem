#pragma once

#include "MatrixBlock.h"

class PoolingLayer {
private:
	unsigned int filter_size;

	MatrixBlock inputs_device;
	MatrixBlock outputs_devices;

public:
	MatrixBlock gradients_device;

	PoolingLayer(const int filter_size, const int gradients_size, const int gradients_depth);
	MatrixBlock& forward(MatrixBlock& input_matrixes, MatrixBlock& prev_gradient_matrixes);
	void backward(MatrixBlock& prev_gradient_matrixes);
	void freeMemory();
};