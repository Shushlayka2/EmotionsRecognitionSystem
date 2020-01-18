#include <iostream>

#include "PoolingLayer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 32

texture<float, 2> InputMatrixesRef;

__global__ void cuda_pooling(float* result, const int cols, const int rows, const int feature_map_matrix_size, const int feature_map_cols, const int filter_size)
{
	int block_x = blockDim.x * blockIdx.x + threadIdx.x; // blockDim.x = 32; blockIdx.x = 0; 
	int block_y = blockDim.y * blockIdx.y + threadIdx.y; // blockDim.y = 32; blockIdx.y = 0;
	int block_z = blockDim.z * blockIdx.z + threadIdx.z; // threadId.z = 0; blockDim.z = 1

	int x = block_x * filter_size; 
	int y = block_y * filter_size;

	if (x < cols && y < rows)
	{
		y *= cols;
		int filter_upper_position = y + x;
		int filter_right_border = __min(x + filter_size - 1, cols - 1) - x;
		int filter_bottom_position = __min(filter_upper_position + cols * (filter_size - 1), x + cols * (rows - 1));

		float max_val = 0.0f;
		for (int i = filter_upper_position; i <= filter_bottom_position; i += cols)
		{
			for (int j = 0; j <= filter_right_border; j++)
			{
				float element = tex2D(InputMatrixesRef, i + j, block_z);
				max_val = __max(element, max_val);
				
				//test
				//printf("%u %u % u: %u %u %u %f\n", j, i, block_z, filter_upper_position, filter_bottom_position, filter_right_border, element);
			}
		}
		result[block_z * feature_map_matrix_size + block_y * feature_map_cols + block_x] = max_val;
		
		//test
		//printf("%u, %u: %f\n", block_x, block_y, result[block_z * feature_map_matrix_size + block_y * feature_map_cols + block_x]);
	}
}

PooingLayer::PooingLayer(MatrixBlock& input_matrixes, const int filter_size) {
	this->input_matrixes = input_matrixes;
	this->filter_size = filter_size;
}

MatrixBlock PooingLayer::get_pooled_feature_map() {
	return pooled_feature_map;
}

void  PooingLayer::forward() {
	float* matrixes_device;
	float* pooled_feature_map_matrix;
	float* pooled_feature_map_device;

	unsigned int pooled_feature_map_depth = input_matrixes.depth;
	unsigned int pooled_feature_map_cols = input_matrixes.cols_count / filter_size + (input_matrixes.cols_count % filter_size == 0 ? 0 : 1);
	unsigned int pooled_feature_map_rows = input_matrixes.rows_count / filter_size + (input_matrixes.rows_count % filter_size == 0 ? 0 : 1);
	unsigned int pooled_feature_map_size = pooled_feature_map_cols * pooled_feature_map_rows;

	rsize_t input_matrixes_pitch;

	pooled_feature_map_matrix = (float*)malloc(sizeof(float) * pooled_feature_map_depth * pooled_feature_map_size);
	cudaMalloc((void**)&pooled_feature_map_device, sizeof(float) * pooled_feature_map_depth * pooled_feature_map_size);
	cudaMallocPitch((void**)&matrixes_device, &input_matrixes_pitch, input_matrixes.matrixes_size * sizeof(float), input_matrixes.depth);
	cudaMemcpy2D(matrixes_device, input_matrixes_pitch, input_matrixes.data, input_matrixes.matrixes_size * sizeof(float), input_matrixes.matrixes_size * sizeof(float), input_matrixes.depth, cudaMemcpyHostToDevice);
	cudaBindTexture2D(0, InputMatrixesRef, matrixes_device, InputMatrixesRef.channelDesc, input_matrixes.matrixes_size, input_matrixes.depth, input_matrixes_pitch);

	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 blocksPerGrid = dim3(pooled_feature_map_cols / BLOCK_SIZE + (pooled_feature_map_cols % BLOCK_SIZE == 0 ? 0 : 1), pooled_feature_map_rows / BLOCK_SIZE + (pooled_feature_map_rows % BLOCK_SIZE == 0 ? 0 : 1), pooled_feature_map_depth);
	cuda_pooling << <blocksPerGrid, threadsPerBlock >> > (pooled_feature_map_device, input_matrixes.cols_count, input_matrixes.rows_count, pooled_feature_map_size, pooled_feature_map_cols, filter_size);
	cudaDeviceSynchronize();

	cudaMemcpy(pooled_feature_map_matrix, pooled_feature_map_device, sizeof(float) * pooled_feature_map_depth * pooled_feature_map_size, cudaMemcpyDeviceToHost);

	pooled_feature_map = MatrixBlock(pooled_feature_map_matrix, pooled_feature_map_rows, pooled_feature_map_cols, pooled_feature_map_depth);

	cudaUnbindTexture(InputMatrixesRef);
	cudaFree(matrixes_device);
	cudaFree(pooled_feature_map_device);
}
