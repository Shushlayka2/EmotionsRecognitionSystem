#pragma once

#include "MatrixBlock.h"

class ConvolutionalLayer {
private:
	MatrixBlock input_matrixes; 
	MatrixBlock filters;
	MatrixBlock feature_map;
public:
	ConvolutionalLayer(MatrixBlock& input_matrixes, const int convolution_filters_half_size, const int filters_count);

	MatrixBlock get_feature_map();

	void convolve();
};