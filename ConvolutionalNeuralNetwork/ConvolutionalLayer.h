#pragma once

#include "MatrixBlock.h"

class ConvolutionalLayer {
private:
	MatrixBlock inputs_device;
	MatrixBlock filters_device;
	MatrixBlock deltas_device;
	MatrixBlock outputs_devices;
	void convolve();
	void activate();

public:
	ConvolutionalLayer(const int filters_size, const int filters_count);
	MatrixBlock& forward(MatrixBlock& input_matrixes);
	void freeMemory();
};