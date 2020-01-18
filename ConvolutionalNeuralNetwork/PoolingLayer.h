#pragma once
#include "MatrixBlock.h"

class PooingLayer {
private:
	MatrixBlock input_matrixes;
	MatrixBlock pooled_feature_map;

	unsigned int filter_size;

public:
	PooingLayer(MatrixBlock& input_matrixes, const int filter_size);

	MatrixBlock get_pooled_feature_map();

	void forward();
};