#include <iostream>

#include "Random.h"
#include "ConvolutionalLayer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

texture<float, 2, cudaReadModeElementType> InputMatrixesRef;
texture<float, 2, cudaReadModeElementType> FiltersRef;

__global__ void cuda_convolve(float** result, const int cols, const int rows, const int depth, const int filter_size)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;

	int fm_cols = cols - filter_size + 1;
	int fm_rows = rows - filter_size + 1;

	if (x < fm_cols && y < fm_rows && z < depth)
	{
		int feature_map_position = y * fm_cols + x;
		result[z][feature_map_position] = 0.0f;
		for (int i = 0; i < filter_size; i++)
		{
			for (int j = 0; j < filter_size; j++)
			{
				int matrix_position = (y + i) * cols + (x + j);
				int filter_position = i * filter_size + j;
				result[z][feature_map_position] += tex2D(InputMatrixesRef, matrix_position, blockIdx.z) *
					tex2D(FiltersRef, filter_position, threadIdx.z);
			}
		}
		//test
		printf("%u %u %u: %f\n", x, y, z, result[z][feature_map_position]);
	}
}

ConvolutionalLayer::ConvolutionalLayer(MatrixBlock& input_matrixes, const int filter_size, const int filters_count) {
	this->input_matrixes = input_matrixes;
	unsigned int filter_elements_count = filter_size * filter_size * filters_count;
	float* random_elements = (float*)malloc(sizeof(float) * filter_elements_count);
	set_normal_random(random_elements, filter_elements_count);
	
	//test
	for (int i = 0; i < filter_elements_count; i++)
		random_elements[i] = i;
	
	filters = MatrixBlock(random_elements, filter_size, filter_size, filters_count);
}

MatrixBlock ConvolutionalLayer::get_feature_map() {
	return feature_map;
}

void ConvolutionalLayer::convolve() {
	float** feature_map_matrix;
	float** matrixes_device;
	float** filters_device;
	float** feature_map_device;
	float** feature_map_host_pointer;
	float** matrixes_host_pointer;
	float** filters_host_pointer;

	unsigned int feature_map_depth = input_matrixes.depth * filters.depth;
	unsigned int feature_map_cols = input_matrixes.cols_count - filters.cols_count + 1;
	unsigned int feature_map_rows = input_matrixes.rows_count - filters.rows_count + 1;
	unsigned int feature_map_size = feature_map_cols * feature_map_rows;
	
	{
		matrixes_host_pointer = (float**)malloc(sizeof(float*) * input_matrixes.depth);
		cudaMalloc((void**)&matrixes_device, sizeof(float*) * input_matrixes.depth);
		for (int i = 0; i < input_matrixes.depth; i++)
		{
			cudaMalloc((void**)&matrixes_host_pointer[i], sizeof(float) * input_matrixes.matrixes_size);
			cudaMemcpy(matrixes_host_pointer[i], input_matrixes.matrixes[i], sizeof(float) * input_matrixes.matrixes_size, cudaMemcpyHostToDevice);
		}
		cudaMemcpy(matrixes_device, matrixes_host_pointer, sizeof(float*) * input_matrixes.depth, cudaMemcpyHostToDevice);

		filters_host_pointer = (float**)malloc(sizeof(float*) * filters.depth);
		cudaMalloc((void**)&filters_device, sizeof(float*) * filters.depth);
		for (int i = 0; i < filters.depth; i++)
		{
			cudaMalloc((void**)&filters_host_pointer[i], sizeof(float) * filters.matrixes_size);
			cudaMemcpy(filters_host_pointer[i], filters.matrixes[i], sizeof(float) * filters.matrixes_size, cudaMemcpyHostToDevice);
		}
		cudaMemcpy(filters_device, filters_host_pointer, sizeof(float*) * filters.depth, cudaMemcpyHostToDevice);

		feature_map_matrix = (float**)malloc(sizeof(float*) * feature_map_depth);
		feature_map_host_pointer = (float**)malloc(sizeof(float*) * feature_map_depth);
		cudaMalloc((void**)&feature_map_device, sizeof(float*) * feature_map_depth);
		for (int i = 0; i < feature_map_depth; i++)
		{
			feature_map_matrix[i] = (float*)malloc(sizeof(float) * feature_map_size);
			cudaMalloc((void**)&feature_map_host_pointer[i], sizeof(float) * feature_map_size);
		}
		cudaMemcpy(feature_map_device, feature_map_host_pointer, sizeof(float*) * feature_map_depth, cudaMemcpyHostToDevice);
	}

	cudaBindTexture2D(NULL, &InputMatrixesRef, matrixes_device, &InputMatrixesRef.channelDesc, input_matrixes.matrixes_size, input_matrixes.depth, sizeof(float) * input_matrixes.matrixes_size);
	cudaBindTexture2D(NULL, &FiltersRef, filters_device, &FiltersRef.channelDesc, filters.matrixes_size, filters.depth, sizeof(float) * filters.matrixes_size);

	dim3 threadsPerBlock = dim3(10, 10, filters.depth);
	dim3 blocksPerGrid = dim3(input_matrixes.cols_count / 10 + (input_matrixes.cols_count % 10 == 0 ? 0 : 1),
		input_matrixes.rows_count / 10 + (input_matrixes.rows_count % 10 == 0 ? 0 : 1),
		input_matrixes.depth);

	cuda_convolve << <blocksPerGrid, threadsPerBlock>> > (feature_map_device, input_matrixes.cols_count, input_matrixes.rows_count, feature_map_depth, filters.cols_count);
	
	for (int i = 0; i < feature_map_depth; i++)
	{
		cudaMemcpy(feature_map_matrix[i], feature_map_host_pointer[i], sizeof(float) * feature_map_size, cudaMemcpyDeviceToHost);
	}
	feature_map = MatrixBlock(feature_map_matrix, feature_map_rows, feature_map_cols, feature_map_depth);
}