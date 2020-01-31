#pragma once

#include "MatrixBlock.h"

class ConvolutionalLayer {
private:
	MatrixBlock inputs_device;
	MatrixBlock filters_device;
	MatrixBlock outputs_devices;
	void convolve();
	void activate();

public:
	MatrixBlock gradients_device;

	ConvolutionalLayer(const int filters_size, const int filters_count, const int gradients_size, const int gradients_depth);
	MatrixBlock& forward(MatrixBlock& input_matrixes);
	void backward(MatrixBlock& prev_gradient_matrixes);
	void correct();
	void freeMemory();
};