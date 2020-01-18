#include "MatrixBlock.h"
#include "CustomException.h"

MatrixBlock::MatrixBlock() : matrixes(nullptr), cols_count(0), rows_count(0), matrixes_size(0), depth(0) {}

MatrixBlock::MatrixBlock(const unsigned int rows_count, const unsigned int cols_count, const unsigned int depth) {
	this->depth = depth;
	this->cols_count = cols_count;
	this->rows_count = rows_count;
	matrixes_size = rows_count * cols_count;
	matrixes = new float* [depth];
	for (int i = 0; i < depth; i++)
	{
		matrixes[i] = new float[matrixes_size];
	}
}

MatrixBlock::MatrixBlock(float* existing_array, const unsigned int rows_count, const unsigned int cols_count, const unsigned int depth) {
	this->depth = depth;
	this->cols_count = cols_count;
	this->rows_count = rows_count;
	matrixes_size = rows_count * cols_count;
	matrixes = new float* [depth];
	for (int i = 0; i < depth; i++)
	{
		matrixes[i] = new float[matrixes_size];
		for (int j = 0; j < matrixes_size; j++)
		{
			matrixes[i][j] = existing_array[i * depth + j];
		}
	}
}

MatrixBlock::MatrixBlock(float** existing_array, const unsigned int rows_count, const unsigned int cols_count, const unsigned int depth) {
	this->depth = depth;
	this->cols_count = cols_count;
	this->rows_count = rows_count;
	matrixes_size = rows_count * cols_count;
	matrixes = existing_array;
}