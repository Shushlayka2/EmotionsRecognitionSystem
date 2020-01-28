#pragma once

#include "MatrixBlock.h"

class PoolingLayer {
private:
	unsigned int filter_size;

	MatrixBlock inputs_device;
	MatrixBlock deltas_device;
	MatrixBlock outputs_devices;

public:
	PoolingLayer(const int filter_size);
	MatrixBlock& forward(MatrixBlock& input_matrixes);
	void freeMemory();
};