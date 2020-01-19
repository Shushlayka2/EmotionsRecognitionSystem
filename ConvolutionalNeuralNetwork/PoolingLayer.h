#pragma once
#include "MatrixBlock.h"

class PooingLayer {
private:
	unsigned int filter_size;

public:
	PooingLayer(const int filter_size);

	MatrixBlock& forward(MatrixBlock& input_matrixes);
};