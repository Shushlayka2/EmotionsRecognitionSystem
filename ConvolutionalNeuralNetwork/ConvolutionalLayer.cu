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

__global__ void cuda_convolve(float* feature_map, float* biases, const int inp_cols, const int fm_cols, const int fm_rows, size_t fm_pitch, const int fm_depth, const int filter_size)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockDim.z * blockIdx.z + threadIdx.z;

	if (x < fm_cols && y < fm_rows && z < fm_depth)
	{
		float* feature_map_matrix_start = (float*)((char*)feature_map + z * fm_pitch);
		int feature_map_position = y * fm_cols + x;
		float sum = biases[z];
		for (int i = 0; i < filter_size; i++)
		{
			for (int j = 0; j < filter_size; j++)
			{
				int matrix_position = (y + i) * inp_cols + (x + j);
				int filter_position = i * filter_size + j;
				sum += tex2D(MatrixesRef, matrix_position, blockIdx.z) * tex2D(FiltersRef, filter_position, threadIdx.z);
			}
		}
		feature_map_matrix_start[feature_map_position] = __max(sum, 0);
	}
}

__global__ void cuda_cross_correlation(float* prev_gradients, const int prev_gr_cols, const int prev_gr_rows, size_t prev_gr_pitch, const int gr_cols, const int gr_rows, const int filter_size, const int filters_count, const int padding)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockIdx.z;

	if (x < prev_gr_cols && y < prev_gr_rows)
	{
		float* prev_gradient_matrix_start = (float*)((char*)prev_gradients + z * prev_gr_pitch);
		int prev_gradient_position = y * prev_gr_cols + x;
		float delta_sum = 0.0f;
		prev_gradient_matrix_start[prev_gradient_position] = 0.0f;

		for (int i = 0; i < filter_size; i++)
		{
			for (int j = 0; j < filter_size; j++)
			{
				int gr_y_pos = y + i - padding;
				int gr_x_pos = x + j - padding;
				float is_inside = gr_y_pos >= 0 || gr_y_pos < gr_rows || gr_x_pos >= 0 || gr_x_pos < gr_cols;

				int matrix_position = gr_y_pos * gr_cols + gr_x_pos;
				int filter_position = ((filter_size - 1) - i) * filter_size + ((filter_size - 1) - j);

				for (int l = 0; l < filters_count; l++)
				{
					delta_sum += is_inside * tex2D(MatrixesRef, matrix_position, z * filters_count + l) * tex2D(FiltersRef, filter_position, l);
				}
			}
		}
		prev_gradient_matrix_start[prev_gradient_position] += delta_sum;
	}
}

__global__ void cuda_correct_filters(float* filters, float* biases, const int fl_cols, const int fl_rows, size_t fl_pitch, const int gr_cols, const int gr_rows, const int in_cols, const int in_count, const int fl_count)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockIdx.z;

	if (x < gr_cols && y < gr_rows)
	{
		float* filter_matrix_start = (float*)((char*)filters + z * fl_pitch);
		int filter_position = y * fl_cols + x;
		float delta_sum = 0.0f;
		for (int l = 0; l < in_count; l++)
		{
			int gr_num = l * fl_count + z;
			int gr_position = y * gr_cols + x;
			for (int i = 0; i < fl_rows; i++)
			{
				for (int j = 0; j < fl_cols; j++)
				{
					int matrix_position = (y + i) * in_cols + (x + j);
					delta_sum += (tex2D(OutputsRef, gr_position, gr_num) > 0.0f)* tex2D(MatrixesRef, matrix_position, l)* tex2D(FiltersRef, gr_position, gr_num);
				}
			}
			biases[gr_num] += tex2D(FiltersRef, gr_position, gr_num);
		}
		filter_matrix_start[filter_position] -= LearningRate * delta_sum;
	}
}

ConvolutionalLayer::ConvolutionalLayer(const int filters_size, const int filters_count, const int outputs_size, const int outputs_depth) {
	
	filters_device = Tensor(filters_size, filters_size, filters_count);
	filters_device.data = set_normal_random(filters_size * filters_size, filters_count, filters_device.pitch);

	filters_gr_device = Tensor(outputs_size, outputs_size, outputs_depth);
	cudaMallocPitch((void**)&filters_gr_device.data, &filters_gr_device.pitch, filters_gr_device.matrixes_size * sizeof(float), filters_gr_device.depth);

	biases_device = set_repeatable_values(outputs_depth, 0.01f);
}

Tensor& ConvolutionalLayer::forward(Tensor& input_matrixes) {
	
	inputs_device = input_matrixes;
	unsigned int feature_map_depth = inputs_device.depth * filters_device.depth;
	unsigned int feature_map_cols = inputs_device.cols_count - filters_device.cols_count + 1;
	unsigned int feature_map_rows = inputs_device.rows_count - filters_device.rows_count + 1;

	outputs_devices = Tensor(feature_map_rows, feature_map_cols, feature_map_depth);
	cudaMallocPitch((void**)&outputs_devices.data, &outputs_devices.pitch, outputs_devices.matrixes_size * sizeof(float), outputs_devices.depth);

	cudaBindTexture2D(0, MatrixesRef, inputs_device.data, MatrixesRef.channelDesc, inputs_device.matrixes_size, inputs_device.depth, inputs_device.pitch);
	cudaBindTexture2D(0, FiltersRef, filters_device.data, FiltersRef.channelDesc, filters_device.matrixes_size, filters_device.depth, filters_device.pitch);

	dim3 threadsPerBlock = dim3(10, 10, filters_device.depth);
	dim3 blocksPerGrid = dim3(outputs_devices.cols_count / 10 + (outputs_devices.cols_count % 10 == 0 ? 0 : 1),
		outputs_devices.rows_count / 10 + (outputs_devices.rows_count % 10 == 0 ? 0 : 1), inputs_device.depth);

	cuda_convolve << <blocksPerGrid, threadsPerBlock >> > (outputs_devices.data, biases_device, inputs_device.cols_count, outputs_devices.cols_count, outputs_devices.rows_count, outputs_devices.pitch, outputs_devices.depth, filters_device.cols_count);
	cudaDeviceSynchronize();
	cudacall(cudaGetLastError());

	cudaUnbindTexture(MatrixesRef);
	cudaUnbindTexture(FiltersRef);

	return outputs_devices;
}

void ConvolutionalLayer::backward(Tensor& prev_gradient_matrixes) {

	cudaBindTexture2D(0, FiltersRef, filters_device.data, FiltersRef.channelDesc, filters_device.matrixes_size, filters_device.depth, filters_device.pitch);
	cudaBindTexture2D(0, MatrixesRef, filters_gr_device.data, MatrixesRef.channelDesc, filters_gr_device.matrixes_size, filters_gr_device.depth, filters_gr_device.pitch);

	dim3 threadsPerBlock = dim3(32, 32, 1);
	dim3 blocksPerGrid = dim3(prev_gradient_matrixes.cols_count / 32 + (prev_gradient_matrixes.cols_count % 32 == 0 ? 0 : 1),
		prev_gradient_matrixes.rows_count / 32 + (prev_gradient_matrixes.rows_count % 32 == 0 ? 0 : 1), prev_gradient_matrixes.depth);
	
	cuda_cross_correlation << <blocksPerGrid, threadsPerBlock >> > (prev_gradient_matrixes.data, prev_gradient_matrixes.cols_count, prev_gradient_matrixes.rows_count, prev_gradient_matrixes.pitch, filters_gr_device.cols_count, filters_gr_device.rows_count, filters_device.cols_count, filters_device.depth, filters_device.cols_count - 1);
	cudaDeviceSynchronize();
	cudacall(cudaGetLastError());

	cudaUnbindTexture(FiltersRef);
	cudaUnbindTexture(MatrixesRef);
}

void ConvolutionalLayer::correct() {
	
	cudaBindTexture2D(0, FiltersRef, filters_gr_device.data, FiltersRef.channelDesc, filters_gr_device.matrixes_size, filters_gr_device.depth, filters_gr_device.pitch);
	cudaBindTexture2D(0, MatrixesRef, inputs_device.data, MatrixesRef.channelDesc, inputs_device.matrixes_size, inputs_device.depth, inputs_device.pitch);
	cudaBindTexture2D(0, OutputsRef, outputs_devices.data, OutputsRef.channelDesc, outputs_devices.matrixes_size, outputs_devices.depth, outputs_devices.pitch);
	
	dim3 threadsPerBlock = dim3(32, 32, 1);
	dim3 blocksPerGrid = dim3(filters_gr_device.cols_count / 32 + (filters_gr_device.cols_count % 32 == 0 ? 0 : 1),
		filters_gr_device.rows_count / 32 + (filters_gr_device.rows_count % 32 == 0 ? 0 : 1), filters_device.depth);
	
	cuda_correct_filters << <blocksPerGrid, threadsPerBlock >> > (filters_device.data, biases_device, filters_device.cols_count, filters_device.rows_count, filters_device.pitch, filters_gr_device.cols_count, filters_gr_device.rows_count, inputs_device.rows_count, inputs_device.depth, filters_device.depth);
	cudaDeviceSynchronize();
	cudacall(cudaGetLastError());

	cudaUnbindTexture(FiltersRef);
	cudaUnbindTexture(MatrixesRef);
	cudaUnbindTexture(OutputsRef);
}

void ConvolutionalLayer::freeMemory() {
	cudaFree(inputs_device.data);
	cudaFree(filters_device.data);
	cudaFree(filters_gr_device.data);
	cudaFree(outputs_devices.data);
}