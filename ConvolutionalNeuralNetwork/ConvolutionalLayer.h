#pragma once

#include "MatrixBlock.h"

class ConvolutionalLayer {
private:
	MatrixBlock filters;
public:
	ConvolutionalLayer(const int filter_size, const int filters_count);

	MatrixBlock& forward(MatrixBlock& input_matrixes);
};