#include "Tensor.h"
#include "CustomException.h"

Tensor::Tensor() : data(nullptr), cols_count(0), rows_count(0), matrixes_size(0), depth(0) {}

Tensor::~Tensor() {}

Tensor::Tensor(const unsigned int rows_count, const unsigned int cols_count, const unsigned int depth, size_t pitch) {
	this->depth = depth;
	this->pitch = pitch;
	this->cols_count = cols_count;
	this->rows_count = rows_count;
	matrixes_size = rows_count * cols_count;
}

Tensor::Tensor(float* existing_array, const unsigned int rows_count, const unsigned int cols_count, const unsigned int depth, size_t pitch) {
	this->depth = depth;
	this->pitch = pitch;
	this->cols_count = cols_count;
	this->rows_count = rows_count;
	matrixes_size = rows_count * cols_count;
	data = existing_array;
}