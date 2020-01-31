#include <iostream>

#include "Random.h"
#include "CustomException.h"
#include "ConvolutionalLayer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define LearningRate 0.0005f

texture<float, 2> MatrixesRef;
texture<float, 2> FiltersRef;
texture<float, 2> OutputsRef;

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
				feature_map_matrix_start[feature_map_position] += tex2D(MatrixesRef, matrix_position, blockIdx.z) * tex2D(FiltersRef, filter_position, threadIdx.z);
			}
		}
	}
}

__global__ void cuda_back_convolve(float* result, const int cols, const int rows, size_t pitch, const int filter_size, const int filters_count)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockIdx.z;

	int gr_cols = cols - filter_size + 1;
	int gr_rows = rows - filter_size + 1;

	if (x < gr_cols && y < gr_rows)
	{
		float* result_start = (float*)((char*)result + z * pitch);
		int result_position = y * gr_cols + x;
		result_start[result_position] = 0.0f;
		for (int i = 0; i < filter_size; i++)
		{
			for (int j = 0; j < filter_size; j++)
			{
				int matrix_position = (y + i) * cols + (x + j);
				int filter_position = i * filter_size + j;
				for (int l = 0; l < filters_count; l++)
				{
					result_start[result_position] += tex2D(MatrixesRef, matrix_position, z * filters_count + l) * tex2D(FiltersRef, filter_position, l);
				}
			}
		}
	}
}

__global__ void cuda_correct_filters(float* result, const int fl_cols, const int fl_rows, size_t pitch, const int gr_cols, const int gr_rows, const int inputs_count, const int filters_count)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockIdx.z;

	if (x < fl_cols && y < fl_rows)
	{
		int in_cols = fl_cols + gr_cols - 1;
		float* result_start = (float*)((char*)result + z * pitch);
		int result_position = y * fl_cols + x;
		for (int i = 0; i < gr_rows; i++)
		{
			for (int j = 0; j < gr_cols; j++)
			{
				int matrix_position = (y + i) * in_cols + (x + j);
				int gr_position = i * gr_cols + j;
				for (int l = 0; l < inputs_count; l++)
				{
					int gr_num = l * filters_count + z;
					result_start[result_position] -= LearningRate * (tex2D(OutputsRef, gr_position, gr_num) > 0.0f) * tex2D(MatrixesRef, matrix_position, l) * tex2D(FiltersRef, gr_position, gr_num);
				}
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

__global__ void cuda_revert_filters(float* filters, float* results, size_t filters_pitch, size_t results_pitch, const int cols, const int rows, const int depth)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;

	if (x < cols && y < rows && z < depth)
	{
		float* filters_start = (float*)((char*)filters + z * filters_pitch);
		float* results_start = (float*)((char*)results + z * results_pitch);
		results_start[(rows - y - 1) * cols + (cols - x - 1)] = filters_start[y * cols + x];
	}
}

__global__ void cuda_add_padding(float* inn_gr, float* out_gr, size_t inn_gr_pitch, size_t out_gr_pitch, const int out_gr_cols, const int out_gr_rows, const int depth, const int border_size, const int inn_gr_cols)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;
	if (x >= border_size && x < out_gr_cols - border_size && y >= border_size && y < out_gr_rows - border_size && z < depth)
	{
		float* inn_gr_start = (float*)((char*)inn_gr + z * inn_gr_pitch);
		float* out_gr_start = (float*)((char*)out_gr + z * out_gr_pitch);
		out_gr_start[y * out_gr_cols + x] = inn_gr_start[(y - border_size) * inn_gr_cols + (x - border_size)];
	}
}

ConvolutionalLayer::ConvolutionalLayer(const int filters_size, const int filters_count, const int gradients_size, const int gradients_depth) {
	size_t pitch;
	float* random_elements;
	random_elements = set_normal_random(filters_size * filters_size, filters_count, pitch);
	filters_device = MatrixBlock(random_elements, filters_size, filters_size, filters_count, pitch);

	gradients_device = MatrixBlock(gradients_size, gradients_size, gradients_depth);
	cudaMallocPitch((void**)&gradients_device.data, &gradients_device.pitch, gradients_device.matrixes_size * sizeof(float), gradients_device.depth);
}

MatrixBlock& ConvolutionalLayer::forward(MatrixBlock& input_matrixes) {
	
	inputs_device = input_matrixes;	
	unsigned int feature_map_depth = inputs_device.depth * filters_device.depth;
	unsigned int feature_map_cols = inputs_device.cols_count - filters_device.cols_count + 1;
	unsigned int feature_map_rows = inputs_device.rows_count - filters_device.rows_count + 1;

	outputs_devices = MatrixBlock(feature_map_rows, feature_map_cols, feature_map_depth);
	cudaMallocPitch((void**)&outputs_devices.data, &outputs_devices.pitch, outputs_devices.matrixes_size * sizeof(float), outputs_devices.depth);

	cudaBindTexture2D(0, MatrixesRef, inputs_device.data, MatrixesRef.channelDesc, inputs_device.matrixes_size, inputs_device.depth, inputs_device.pitch);
	cudaBindTexture2D(0, FiltersRef, filters_device.data, FiltersRef.channelDesc, filters_device.matrixes_size, filters_device.depth, filters_device.pitch);

	convolve();
	activate();

	cudaUnbindTexture(MatrixesRef);
	cudaUnbindTexture(FiltersRef);

	return outputs_devices;
}

void ConvolutionalLayer::backward(MatrixBlock& prev_gradient_matrixes) {

	size_t reverted_filters_pitch, padded_gradients_pitch;
	float* padded_gradients_device;
	float* reverted_filters_device;
	int border_size = (filters_device.cols_count - 1);
	int padded_gradients_matrixes_cols = gradients_device.cols_count + border_size * 2;
	int padded_gradients_matrixes_rows = gradients_device.rows_count + border_size * 2;
	int padded_gradients_matrixes_size = padded_gradients_matrixes_cols * padded_gradients_matrixes_rows;
	cudaMallocPitch((void**)&padded_gradients_device, &padded_gradients_pitch, padded_gradients_matrixes_size * sizeof(float), gradients_device.depth);
	cudaMallocPitch((void**)&reverted_filters_device, &reverted_filters_pitch, filters_device.matrixes_size * sizeof(float), filters_device.depth);
	
	dim3 threadsPerBlock = dim3(10, 10, 10);
	dim3 blocksPerGrid = dim3(filters_device.cols_count / 10 + (filters_device.cols_count % 10 == 0 ? 0 : 1),
		filters_device.rows_count / 10 + (filters_device.rows_count % 10 == 0 ? 0 : 1),
		filters_device.depth / 10 + (filters_device.depth % 10 == 0 ? 0 : 1));

	cuda_revert_filters << <blocksPerGrid, threadsPerBlock >> > (filters_device.data, reverted_filters_device, filters_device.pitch, reverted_filters_pitch, filters_device.cols_count, filters_device.rows_count, filters_device.depth);
	cudaDeviceSynchronize();
	cudacall(cudaGetLastError());

	blocksPerGrid = dim3(padded_gradients_matrixes_cols / 10 + (padded_gradients_matrixes_cols % 10 == 0 ? 0 : 1),
		padded_gradients_matrixes_rows / 10 + (padded_gradients_matrixes_rows % 10 == 0 ? 0 : 1),
		gradients_device.depth / 10 + (gradients_device.depth % 10 == 0 ? 0 : 1));

	cuda_add_padding << <blocksPerGrid, threadsPerBlock >> > (gradients_device.data, padded_gradients_device, gradients_device.pitch, padded_gradients_pitch, padded_gradients_matrixes_cols, padded_gradients_matrixes_rows, gradients_device.depth, border_size, gradients_device.cols_count);
	cudaDeviceSynchronize();
	cudacall(cudaGetLastError());

	cudaBindTexture2D(0, FiltersRef, reverted_filters_device, FiltersRef.channelDesc, filters_device.matrixes_size, filters_device.depth, reverted_filters_pitch);
	cudaBindTexture2D(0, MatrixesRef, padded_gradients_device, MatrixesRef.channelDesc, padded_gradients_matrixes_size, gradients_device.depth, padded_gradients_pitch);

	threadsPerBlock = dim3(10, 10, 1);
	blocksPerGrid = dim3(prev_gradient_matrixes.cols_count / 10 + (prev_gradient_matrixes.cols_count % 10 == 0 ? 0 : 1),
		prev_gradient_matrixes.rows_count / 10 + (prev_gradient_matrixes.rows_count % 10 == 0 ? 0 : 1), prev_gradient_matrixes.depth);
	
	cuda_back_convolve << <blocksPerGrid, threadsPerBlock >> > (prev_gradient_matrixes.data, padded_gradients_matrixes_cols, padded_gradients_matrixes_rows, padded_gradients_pitch, filters_device.cols_count, filters_device.depth);
	cudaDeviceSynchronize();
	cudacall(cudaGetLastError());

	cudaUnbindTexture(FiltersRef);
	cudaUnbindTexture(MatrixesRef);
	cudaFree(reverted_filters_device);
}

void ConvolutionalLayer::correct() {
	
	cudaBindTexture2D(0, FiltersRef, gradients_device.data, FiltersRef.channelDesc, gradients_device.matrixes_size, gradients_device.depth, gradients_device.pitch);
	cudaBindTexture2D(0, MatrixesRef, inputs_device.data, MatrixesRef.channelDesc, inputs_device.matrixes_size, inputs_device.depth, inputs_device.pitch);
	cudaBindTexture2D(0, OutputsRef, outputs_devices.data, OutputsRef.channelDesc, outputs_devices.matrixes_size, outputs_devices.depth, outputs_devices.pitch);
	
	dim3 threadsPerBlock = dim3(10, 10, 1);
	dim3 blocksPerGrid = dim3(filters_device.cols_count / 10 + (filters_device.cols_count % 10 == 0 ? 0 : 1),
		filters_device.rows_count / 10 + (filters_device.rows_count % 10 == 0 ? 0 : 1), filters_device.depth);
	
	cuda_correct_filters << <blocksPerGrid, threadsPerBlock >> > (filters_device.data, filters_device.cols_count, filters_device.rows_count, filters_device.pitch, gradients_device.cols_count, gradients_device.rows_count, inputs_device.depth, filters_device.depth);
	cudaDeviceSynchronize();
	cudacall(cudaGetLastError());

	cudaUnbindTexture(FiltersRef);
	cudaUnbindTexture(MatrixesRef);
	cudaUnbindTexture(OutputsRef);
}

void ConvolutionalLayer::convolve() {

	dim3 threadsPerBlock = dim3(10, 10, filters_device.depth);
	dim3 blocksPerGrid = dim3(outputs_devices.cols_count / 10 + (outputs_devices.cols_count % 10 == 0 ? 0 : 1),
		outputs_devices.rows_count / 10 + (outputs_devices.rows_count % 10 == 0 ? 0 : 1), inputs_device.depth);

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
	cudaFree(gradients_device.data);
	cudaFree(outputs_devices.data);
}