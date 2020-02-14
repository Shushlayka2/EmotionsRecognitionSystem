#include <iostream>

#include "Hub.h"
#include "Random.h"
#include "CustomException.h"
#include "ConvolutionalLayer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 256
#define DOUBLE_BLOCK_SIZE 32
#define LearningRate 0.0005f
#define MAX_ELEMENTS_COUNT 1024
#define SUM_STRIDE 512

texture<float, 2> MatrixesRef;
texture<float, 2> FiltersRef;
texture<float, 2> OutputsRef;

__global__ void cuda_convolve(float* feature_map, float* biases, const int inp_cols, const int inp_depth, const int fm_cols, const int fm_rows, size_t fm_pitch, const int filter_size)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	int z = blockIdx.z;

	if (x < fm_cols && y < fm_rows)
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
				for (int l = 0; l < inp_depth; l++)
					sum += tex2D(MatrixesRef, matrix_position, l) * tex2D(FiltersRef, filter_position, z * inp_depth + l);
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
				float is_inside = (gr_y_pos >= 0 && gr_y_pos < gr_rows && gr_x_pos >= 0 && gr_x_pos < gr_cols);

				int matrix_position = gr_y_pos * gr_cols + gr_x_pos;
				int filter_position = (padding - i) * filter_size + (padding - j);

				for (int l = 0; l < filters_count; l++)
				{
					delta_sum += is_inside * tex2D(MatrixesRef, matrix_position, l) * tex2D(FiltersRef, filter_position, l * filters_count + z);
				}
			}
		}
		prev_gradient_matrix_start[prev_gradient_position] += delta_sum;
	}
}

__global__ void cuda_correct_filters(float* filters, const int fl_size, size_t fl_pitch, const int gr_cols, const int gr_rows, const int gr_count, const int in_cols, const int in_count)
{
	int x = threadIdx.x;
	int y = threadIdx.y;
	int x_block = blockIdx.x;
	int y_block = blockIdx.y;
	int z_block = blockIdx.z;

	int gr_num = z_block / in_count;
	int in_num = z_block / gr_count;

	extern __shared__ float s_deltas[];
	s_deltas[y * blockDim.x + x] = 0.0f;

	int gr_position = y * gr_rows + x;
	int filter_position = y_block * fl_size + x_block;

	if (x < gr_cols && y < gr_rows)
	{
		int matrix_position = (y + y_block) * in_cols + (x + x_block);
		float* filter_matrix_start = (float*)((char*)filters + z_block * fl_pitch);

		s_deltas[gr_position] = (tex2D(OutputsRef, gr_position, gr_num) > 0.0f)* tex2D(MatrixesRef, matrix_position, in_num)* tex2D(FiltersRef, gr_position, gr_num);
	}

	__syncthreads();
	
	for (unsigned int s = SUM_STRIDE; s > 0; s >>= 1) {
		if (gr_position < s)
			s_deltas[gr_position] += s_deltas[gr_position + s];
		__syncthreads();
	}
	if (gr_position == 0)
	{
		float* filter_matrix_start = (float*)((char*)filters + z_block * fl_pitch);
		filter_matrix_start[filter_position] -= LearningRate * s_deltas[0];
	}
}

__global__ void cuda_correct_biases(float* biases, const int b_size, const int gr_cols, const int gr_rows)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x < b_size)
	{
		float sum = 0.0f;
		for (int i = 0; i < gr_rows; i++)
		{
			for (int j = 0; j < gr_cols; j++)
			{
				sum += tex2D(FiltersRef, i * gr_cols + j, x);
			}
		}
		biases[x] -= LearningRate * sum;
	}
}

ConvolutionalLayer::ConvolutionalLayer(const int filters_size, const int filters_count, const int inputs_depth, const int outputs_size, Hub& params_storage) {
	
	int filter_depth = inputs_depth * filters_count;
	filters_device = Tensor(filters_size, filters_size, filter_depth);
	gradients_device = Tensor(outputs_size, outputs_size, filters_count);
	cudaMallocPitch((void**)&gradients_device.data, &gradients_device.pitch, gradients_device.matrixes_size * sizeof(float), gradients_device.depth);

	outputs_devices = Tensor(outputs_size, outputs_size, filters_count);
	cudaMallocPitch((void**)&outputs_devices.data, &outputs_devices.pitch, outputs_devices.matrixes_size * sizeof(float), outputs_devices.depth);

	if (params_storage.get_status() == Status::Training)
	{
		filters_device.data = set_normal_random(filters_size * filters_size, filter_depth, filters_device.pitch);
		biases_device = set_repeatable_values(filters_count, 0.01f);
	}
	else
	{
		params_storage.get_params(filters_device);
		biases_device = params_storage.get_params(filters_count);
	}
}

Tensor& ConvolutionalLayer::forward(Tensor& input_matrixes) {
	
	inputs_device = input_matrixes;

	cudaBindTexture2D(0, MatrixesRef, inputs_device.data, MatrixesRef.channelDesc, inputs_device.matrixes_size, inputs_device.depth, inputs_device.pitch);
	cudaBindTexture2D(0, FiltersRef, filters_device.data, FiltersRef.channelDesc, filters_device.matrixes_size, filters_device.depth, filters_device.pitch);

	dim3 threadsPerBlock = dim3(DOUBLE_BLOCK_SIZE, DOUBLE_BLOCK_SIZE, 1);
	dim3 blocksPerGrid = dim3(outputs_devices.cols_count / DOUBLE_BLOCK_SIZE + (outputs_devices.cols_count % DOUBLE_BLOCK_SIZE == 0 ? 0 : 1),
		outputs_devices.rows_count / DOUBLE_BLOCK_SIZE + (outputs_devices.rows_count % DOUBLE_BLOCK_SIZE == 0 ? 0 : 1), outputs_devices.depth);

	cuda_convolve << <blocksPerGrid, threadsPerBlock >> > (outputs_devices.data, biases_device, inputs_device.cols_count, inputs_device.depth, outputs_devices.cols_count, outputs_devices.rows_count, outputs_devices.pitch, filters_device.cols_count);
	cudacall(cudaGetLastError());

	cudaUnbindTexture(MatrixesRef);
	cudaUnbindTexture(FiltersRef);

	return outputs_devices;
}

void ConvolutionalLayer::backward(Tensor& prev_gradient_matrixes) {

	cudaBindTexture2D(0, FiltersRef, filters_device.data, FiltersRef.channelDesc, filters_device.matrixes_size, filters_device.depth, filters_device.pitch);
	cudaBindTexture2D(0, MatrixesRef, gradients_device.data, MatrixesRef.channelDesc, gradients_device.matrixes_size, gradients_device.depth, gradients_device.pitch);

	dim3 threadsPerBlock = dim3(DOUBLE_BLOCK_SIZE, DOUBLE_BLOCK_SIZE, 1);
	dim3 blocksPerGrid = dim3(prev_gradient_matrixes.cols_count / DOUBLE_BLOCK_SIZE + (prev_gradient_matrixes.cols_count % DOUBLE_BLOCK_SIZE == 0 ? 0 : 1),
		prev_gradient_matrixes.rows_count / DOUBLE_BLOCK_SIZE + (prev_gradient_matrixes.rows_count % DOUBLE_BLOCK_SIZE == 0 ? 0 : 1), prev_gradient_matrixes.depth);
	
	cuda_cross_correlation << <blocksPerGrid, threadsPerBlock >> > (prev_gradient_matrixes.data, prev_gradient_matrixes.cols_count, prev_gradient_matrixes.rows_count, prev_gradient_matrixes.pitch, gradients_device.cols_count, gradients_device.rows_count, filters_device.cols_count, gradients_device.depth, filters_device.cols_count - 1);
	cudacall(cudaGetLastError());

	cudaUnbindTexture(FiltersRef);
	cudaUnbindTexture(MatrixesRef);
}

void ConvolutionalLayer::correct() {

	cudaBindTexture2D(0, FiltersRef, gradients_device.data, FiltersRef.channelDesc, gradients_device.matrixes_size, gradients_device.depth, gradients_device.pitch);
	cudaBindTexture2D(0, MatrixesRef, inputs_device.data, MatrixesRef.channelDesc, inputs_device.matrixes_size, inputs_device.depth, inputs_device.pitch);
	cudaBindTexture2D(0, OutputsRef, outputs_devices.data, OutputsRef.channelDesc, outputs_devices.matrixes_size, outputs_devices.depth, outputs_devices.pitch);
	
	//It can be forced for larger images. This implementation appropriate only for MNIST.
	dim3 threadsPerBlock = dim3(DOUBLE_BLOCK_SIZE, DOUBLE_BLOCK_SIZE, 1);
	dim3 blocksPerGrid = dim3(filters_device.cols_count, filters_device.rows_count, filters_device.depth);

	cuda_correct_filters << <blocksPerGrid, threadsPerBlock, MAX_ELEMENTS_COUNT * sizeof(float) >> > (filters_device.data, filters_device.cols_count, filters_device.pitch, gradients_device.cols_count, gradients_device.rows_count, gradients_device.depth, inputs_device.cols_count, inputs_device.depth);
	cudacall(cudaGetLastError())

	threadsPerBlock = BLOCK_SIZE;
	blocksPerGrid = dim3(gradients_device.depth / BLOCK_SIZE + (gradients_device.depth % BLOCK_SIZE == 0 ? 0 : 1));
	
	cuda_correct_biases << <blocksPerGrid, threadsPerBlock >> > (biases_device, gradients_device.depth, gradients_device.cols_count, gradients_device.rows_count);
	cudacall(cudaGetLastError());

	cudaUnbindTexture(FiltersRef);
	cudaUnbindTexture(MatrixesRef);
	cudaUnbindTexture(OutputsRef);
}

void ConvolutionalLayer::save_params(Hub& params_storage) {

	params_storage.set_params(filters_device);
	params_storage.set_params(biases_device, outputs_devices.depth);
}

void ConvolutionalLayer::freeMemory() {

	cudaFree(filters_device.data);
	cudaFree(gradients_device.data);
	cudaFree(inputs_device.data);
	cudaFree(biases_device);
}