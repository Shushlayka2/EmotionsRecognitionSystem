#include <iostream>

#include "PoolingLayer.h"
#include "CustomException.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 32

texture<float, 2> InputMatrixesRef;

__global__ void cuda_pooling(float* result, const int cols, const int rows, size_t pitch, const int feature_map_cols, const int filter_size)
{
	int block_x = blockDim.x * blockIdx.x + threadIdx.x;
	int block_y = blockDim.y * blockIdx.y + threadIdx.y;
	int block_z = blockDim.z * blockIdx.z + threadIdx.z;

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
			}
		}
		float* feature_map_matrix_start = (float*)((char*)result + block_z * pitch);
		int feature_map_position = block_y * feature_map_cols + block_x;

		feature_map_matrix_start[feature_map_position] = max_val;
	}
}

PoolingLayer::PoolingLayer(const int filter_size) {
	this->filter_size = filter_size;
}

MatrixBlock& PoolingLayer::forward(MatrixBlock& input_matrixes) {
	
	inputs_device = input_matrixes;
	unsigned int pooled_feature_map_cols = inputs_device.cols_count / filter_size + (inputs_device.cols_count % filter_size == 0 ? 0 : 1);
	unsigned int pooled_feature_map_rows = inputs_device.rows_count / filter_size + (inputs_device.rows_count % filter_size == 0 ? 0 : 1);

	outputs_devices = MatrixBlock(pooled_feature_map_rows, pooled_feature_map_cols, inputs_device.depth);
	cudaMallocPitch((void**)&outputs_devices.data, &outputs_devices.pitch, outputs_devices.matrixes_size * sizeof(float), outputs_devices.depth);

	cudaBindTexture2D(0, InputMatrixesRef, inputs_device.data, InputMatrixesRef.channelDesc, inputs_device.matrixes_size, inputs_device.depth, inputs_device.pitch);
	
	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 blocksPerGrid = dim3(pooled_feature_map_cols / BLOCK_SIZE + (pooled_feature_map_cols % BLOCK_SIZE == 0 ? 0 : 1), pooled_feature_map_rows / BLOCK_SIZE + (pooled_feature_map_rows % BLOCK_SIZE == 0 ? 0 : 1), inputs_device.depth);
	cuda_pooling << <blocksPerGrid, threadsPerBlock >> > (outputs_devices.data, input_matrixes.cols_count, input_matrixes.rows_count, outputs_devices.pitch, outputs_devices.cols_count, filter_size);
	cudaDeviceSynchronize();
	cudacall(cudaGetLastError());

	cudaUnbindTexture(InputMatrixesRef);

	return outputs_devices;
}

void PoolingLayer::freeMemory() {
	cudaFree(inputs_device.data);
	//cudaFree(deltas_device.data);
	cudaFree(outputs_devices.data);
}
