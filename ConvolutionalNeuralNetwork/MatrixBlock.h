#pragma once

struct MatrixBlock
{
	float* data;
	unsigned int rows_count;
	unsigned int cols_count;
	unsigned int matrixes_size;
	unsigned int depth;

	MatrixBlock();

	MatrixBlock(const unsigned int rows_count, const unsigned int cols_count, const unsigned int depth);

	MatrixBlock(float* existing_array, const unsigned int rows_count, const unsigned int cols_count, const unsigned int depth);
};