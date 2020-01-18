#include <iostream>

#include "Random.h"
#include "ConvolutionalLayer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

texture<float, 2> InputMatrixesRef;
texture<float, 2> FiltersRef;

__global__ void cuda_convolve(float* result, const int cols, const int rows, const int matrixes_size, const int depth, const int filter_size)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;

	int fm_cols = cols - filter_size + 1;
	int fm_rows = rows - filter_size + 1;

	if (x < fm_cols && y < fm_rows && z < depth)
	{
		int feature_map_position = z * matrixes_size + y * fm_cols + x;
		result[feature_map_position] = 0.0f;
		for (int i = 0; i < filter_size; i++)
		{
			for (int j = 0; j < filter_size; j++)
			{
				int matrix_position = (y + i) * cols + (x + j);
				int filter_position = i * filter_size + j;

				//test
				/*float temp1 = tex2D(InputMatrixesRef, matrix_position, blockIdx.z);
				float temp2 = tex2D(FiltersRef, filter_position, threadIdx.z);
				printf("%u %u %u %u %u: %f %f\n", x, y, z, i, j, temp1, temp2);
				float temp = tex2D(InputMatrixesRef, matrix_position, blockIdx.z) * tex2D(FiltersRef, filter_position, threadIdx.z);
				printf("%u %u %u %u %u: %f\n", x, y, z, i, j, temp);*/

				result[feature_map_position] += tex2D(InputMatrixesRef, matrix_position, blockIdx.z) * tex2D(FiltersRef, filter_position, threadIdx.z);
			}
		}
		//test
		//printf("%u %u %u: %f\n", x, y, z, result[feature_map_position]);
	}
}

__global__ void cuda_ReLU(float* arr, const int cols, const int rows, const int matrixes_size, const int depth)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;

	if (x < cols && y < rows && z < depth)
	{
		int feature_map_position = z * matrixes_size + y * cols + x;
		arr[feature_map_position] = __max(arr[feature_map_position], 0);
	}
}

ConvolutionalLayer::ConvolutionalLayer(MatrixBlock& input_matrixes, const int filter_size, const int filters_count) {
	this->input_matrixes = input_matrixes;
	unsigned int filter_elements_count = filter_size * filter_size * filters_count;
	float* random_elements = (float*)malloc(sizeof(float) * filter_elements_count);
	set_normal_random(random_elements, filter_elements_count);

	//test
	/*for (int i = 0; i < filter_elements_count; i++)
		random_elements[i] = i % 9;*/

	filters = MatrixBlock(random_elements, filter_size, filter_size, filters_count);
}

MatrixBlock ConvolutionalLayer::get_feature_map() {
	return feature_map;
}

void ConvolutionalLayer::forward() {
	float* matrixes_device;
	float* filters_device;
	float* feature_map_device;
	float* feature_map_matrix;

	unsigned int feature_map_depth = input_matrixes.depth * filters.depth;
	unsigned int feature_map_cols = input_matrixes.cols_count - filters.cols_count + 1;
	unsigned int feature_map_rows = input_matrixes.rows_count - filters.rows_count + 1;
	unsigned int feature_map_size = feature_map_cols * feature_map_rows;

	feature_map_matrix = (float*)malloc(sizeof(float) * feature_map_depth * feature_map_size);
	cudaMalloc((void**)&feature_map_device, sizeof(float) * feature_map_depth * feature_map_size);

	rsize_t input_matrixes_pitch, filters_pitch;

	cudaMallocPitch((void**)&matrixes_device, &input_matrixes_pitch, input_matrixes.matrixes_size * sizeof(float), input_matrixes.depth);
	cudaMemcpy2D(matrixes_device, input_matrixes_pitch, input_matrixes.data, input_matrixes.matrixes_size * sizeof(float), input_matrixes.matrixes_size * sizeof(float), input_matrixes.depth, cudaMemcpyHostToDevice);
	cudaBindTexture2D(0, InputMatrixesRef, matrixes_device, InputMatrixesRef.channelDesc, input_matrixes.matrixes_size, input_matrixes.depth, input_matrixes_pitch);

	cudaMallocPitch((void**)&filters_device, &filters_pitch, filters.matrixes_size * sizeof(float), filters.depth);
	cudaMemcpy2D(filters_device, filters_pitch, filters.data, filters.matrixes_size * sizeof(float), filters.matrixes_size * sizeof(float), filters.depth, cudaMemcpyHostToDevice);
	cudaBindTexture2D(0, FiltersRef, filters_device, FiltersRef.channelDesc, filters.matrixes_size, filters.depth, filters_pitch);

	dim3 threadsPerBlock = dim3(10, 10, filters.depth);
	dim3 blocksPerGrid = dim3(input_matrixes.cols_count / 10 + (input_matrixes.cols_count % 10 == 0 ? 0 : 1),
		input_matrixes.rows_count / 10 + (input_matrixes.rows_count % 10 == 0 ? 0 : 1),
		input_matrixes.depth);

	cuda_convolve << <blocksPerGrid, threadsPerBlock >> > (feature_map_device, input_matrixes.cols_count, input_matrixes.rows_count, feature_map_size, feature_map_depth, filters.cols_count);
	cudaDeviceSynchronize();

	threadsPerBlock = dim3(10, 10, 10);
	blocksPerGrid = dim3(feature_map_cols / 10 + (feature_map_cols % 10 == 0 ? 0 : 1),
		feature_map_rows / 10 + (feature_map_rows % 10 == 0 ? 0 : 1),
		feature_map_depth / 10 + (feature_map_depth % 10 == 0 ? 0 : 1));

	cuda_ReLU << <blocksPerGrid, threadsPerBlock >> > (feature_map_device, feature_map_cols, feature_map_rows, feature_map_size, feature_map_depth);
	cudaDeviceSynchronize();

	cudaMemcpy(feature_map_matrix, feature_map_device, sizeof(float) * feature_map_size * feature_map_depth, cudaMemcpyDeviceToHost);

	feature_map = MatrixBlock(feature_map_matrix, feature_map_rows, feature_map_cols, feature_map_depth);

	cudaUnbindTexture(InputMatrixesRef);
	cudaUnbindTexture(FiltersRef);
	cudaFree(matrixes_device);
	cudaFree(filters_device);
	cudaFree(feature_map_device);
}