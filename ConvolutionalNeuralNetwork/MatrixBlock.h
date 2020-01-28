#pragma once

struct MatrixBlock
{
	float* data;
	size_t pitch;
	unsigned int rows_count;
	unsigned int cols_count;
	unsigned int matrixes_size;
	unsigned int depth;

	MatrixBlock();
	MatrixBlock(const unsigned int rows_count, const unsigned int cols_count, const unsigned int depth, size_t pitch = 0);
	MatrixBlock(float* existing_array, const unsigned int rows_count, const unsigned int cols_count, const unsigned int depth, size_t pitch = 0);
};