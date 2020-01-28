#include <iostream>

#include "Random.h"
#include "CustomException.h"
#include "ConvolutionalLayer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

texture<float, 2> InputMatrixesRef;
texture<float, 2> FiltersRef;

__global__ void cuda_convolve(float* result, const int cols, const int rows, size_t pitch, const int depth, const int filter_size)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;

	int fm_cols = cols - filter_size + 1;
	int fm_rows = rows - filter_size + 1;

	if (x < fm_cols && y < fm_rows && z < depth)
	{
		float* feature_map_matrix_start = (float*)((char*)result + z * pitch);
		int feature_map_position = y * fm_cols + x;
		feature_map_matrix_start[feature_map_position] = 0.0f;
		for (int i = 0; i < filter_size; i++)
		{
			for (int j = 0; j < filter_size; j++)
			{
				int matrix_position = (y + i) * cols + (x + j);
				int filter_position = i * filter_size + j;
				feature_map_matrix_start[feature_map_position] += tex2D(InputMatrixesRef, matrix_position, blockIdx.z) * tex2D(FiltersRef, filter_position, threadIdx.z);
			}
		}
	}
}

__global__ void cuda_ReLU(float* arr, const int cols, const int rows, size_t pitch, const int depth)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;

	if (x < cols && y < rows && z < depth)
	{
		float* feature_map_matrix_start = (float*)((char*)arr + z * pitch);
		int feature_map_position = y * cols + x;
		feature_map_matrix_start[feature_map_position] = __max(feature_map_matrix_start[feature_map_position], 0);
	}
}

ConvolutionalLayer::ConvolutionalLayer(const int filters_size, const int filters_count) {
	size_t pitch;
	float* random_elements;
	random_elements = set_normal_random(filters_size * filters_size, filters_count, pitch);
	filters_device = MatrixBlock(random_elements, filters_size, filters_size, filters_count, pitch);
}

MatrixBlock& ConvolutionalLayer::forward(MatrixBlock& input_matrixes) {
	
	inputs_device = input_matrixes;	
	unsigned int feature_map_depth = inputs_device.depth * filters_device.depth;
	unsigned int feature_map_cols = inputs_device.cols_count - filters_device.cols_count + 1;
	unsigned int feature_map_rows = inputs_device.rows_count - filters_device.rows_count + 1;

	outputs_devices = MatrixBlock(feature_map_rows, feature_map_cols, feature_map_depth);
	cudaMallocPitch((void**)&outputs_devices.data, &outputs_devices.pitch, outputs_devices.matrixes_size * sizeof(float), outputs_devices.depth);

	cudaBindTexture2D(0, InputMatrixesRef, inputs_device.data, InputMatrixesRef.channelDesc, inputs_device.matrixes_size, inputs_device.depth, inputs_device.pitch);
	cudaBindTexture2D(0, FiltersRef, filters_device.data, FiltersRef.channelDesc, filters_device.matrixes_size, filters_device.depth, filters_device.pitch);

	convolve();
	activate();

	cudaUnbindTexture(InputMatrixesRef);
	cudaUnbindTexture(FiltersRef);

	return outputs_devices;
}

void ConvolutionalLayer::convolve() {

	dim3 threadsPerBlock = dim3(10, 10, filters_device.depth);
	dim3 blocksPerGrid = dim3(inputs_device.cols_count / 10 + (inputs_device.cols_count % 10 == 0 ? 0 : 1),
		inputs_device.rows_count / 10 + (inputs_device.rows_count % 10 == 0 ? 0 : 1),
		inputs_device.depth);

	cuda_convolve << <blocksPerGrid, threadsPerBlock >> > (outputs_devices.data, inputs_device.cols_count, inputs_device.rows_count, outputs_devices.pitch, outputs_devices.depth, filters_device.cols_count);
	cudaDeviceSynchronize();
	cudacall(cudaGetLastError());
}

void ConvolutionalLayer::activate() {
	
	dim3 threadsPerBlock = dim3(10, 10, 10);
	dim3 blocksPerGrid = dim3(outputs_devices.cols_count / 10 + (outputs_devices.cols_count % 10 == 0 ? 0 : 1),
		outputs_devices.rows_count / 10 + (outputs_devices.rows_count % 10 == 0 ? 0 : 1),
		outputs_devices.depth / 10 + (outputs_devices.depth % 10 == 0 ? 0 : 1));

	cuda_ReLU << <blocksPerGrid, threadsPerBlock >> > (outputs_devices.data, outputs_devices.cols_count, outputs_devices.rows_count, outputs_devices.pitch, outputs_devices.depth);
	cudaDeviceSynchronize();
	cudacall(cudaGetLastError());
}

void ConvolutionalLayer::freeMemory() {
	cudaFree(inputs_device.data);
	cudaFree(filters_device.data);
	//cudaFree(deltas_device.data);
	cudaFree(outputs_devices.data);
}