#include <iostream>

#include "PoolingLayer.h"
#include "CustomException.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BLOCK_SIZE 32

texture<float, 2> InputMatrixesRef;
texture<float, 2> GradientMatrixesRef;

__global__ void cuda_pooling(float* result, float* prev_gradients, size_t gr_pitch, const int cols, const int rows, size_t fm_pitch, const int feature_map_cols, const int filter_size)
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

		float* prev_gradients_start = (float*)((char*)prev_gradients + block_z * gr_pitch);

		float max_val = _I32_MIN;
		int max_i, max_j;
		for (int i = filter_upper_position; i <= filter_bottom_position; i += cols)
		{
			for (int j = 0; j <= filter_right_border; j++)
			{
				float element = tex2D(InputMatrixesRef, i + j, block_z);
				(element > max_val) ? (max_val = element, max_i = i, max_j = j) : (max_val);
				prev_gradients_start[i + j] = 0;
			}
		}

		prev_gradients_start[max_i + max_j] = 1.0f;
		float* feature_map_matrix_start = (float*)((char*)result + block_z * fm_pitch);
		int feature_map_position = block_y * feature_map_cols + block_x;

		feature_map_matrix_start[feature_map_position] = max_val;
	}
}

__global__ void cuda_generate_gradients(float* prev_gradients, size_t prev_gr_pitch, const int cols, const int rows, const int cur_gr_cols, const int filter_size)
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
		float* prev_gradients_start = (float*)((char*)prev_gradients + block_z * prev_gr_pitch);

		float element = tex2D(GradientMatrixesRef, block_y * cur_gr_cols + block_x, block_z);
		for (int i = filter_upper_position; i <= filter_bottom_position; i += cols)
		{
			for (int j = 0; j <= filter_right_border; j++)
			{	
				prev_gradients_start[i + j] *= element;
			}
		}
	}
}

PoolingLayer::PoolingLayer(const int filter_size, const int outputs_size, const int outputs_depth) {
	
	this->filter_size = filter_size;
	gradients_device = Tensor(outputs_size, outputs_size, outputs_depth);
	cudaMallocPitch((void**)&gradients_device.data, &gradients_device.pitch, gradients_device.matrixes_size * sizeof(float), gradients_device.depth);

	outputs_devices = Tensor(outputs_size, outputs_size, outputs_depth);
	cudaMallocPitch((void**)&outputs_devices.data, &outputs_devices.pitch, outputs_devices.matrixes_size * sizeof(float), outputs_devices.depth);
}

Tensor& PoolingLayer::forward(Tensor& input_matrixes, Tensor& prev_gradient_matrixes) {
	
	inputs_device = input_matrixes;

	cudaBindTexture2D(0, InputMatrixesRef, inputs_device.data, InputMatrixesRef.channelDesc, inputs_device.matrixes_size, inputs_device.depth, inputs_device.pitch);
	
	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 blocksPerGrid = dim3(outputs_devices.cols_count / BLOCK_SIZE + (outputs_devices.cols_count % BLOCK_SIZE == 0 ? 0 : 1), outputs_devices.rows_count / BLOCK_SIZE + (outputs_devices.rows_count % BLOCK_SIZE == 0 ? 0 : 1), outputs_devices.depth);
	cuda_pooling << <blocksPerGrid, threadsPerBlock >> > (outputs_devices.data, prev_gradient_matrixes.data, prev_gradient_matrixes.pitch, input_matrixes.cols_count, input_matrixes.rows_count, outputs_devices.pitch, outputs_devices.cols_count, filter_size);
	cudacall(cudaGetLastError());

	cudaUnbindTexture(InputMatrixesRef);

	return outputs_devices;
}

void PoolingLayer::backward(Tensor& prev_gradient_matrixes) {

	dim3 threadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
	dim3 blocksPerGrid = dim3(gradients_device.cols_count / BLOCK_SIZE + (gradients_device.cols_count % BLOCK_SIZE == 0 ? 0 : 1), gradients_device.rows_count / BLOCK_SIZE + (gradients_device.rows_count % BLOCK_SIZE == 0 ? 0 : 1), gradients_device.depth);
	cudaBindTexture2D(0, GradientMatrixesRef, gradients_device.data, GradientMatrixesRef.channelDesc, gradients_device.matrixes_size, gradients_device.depth, gradients_device.pitch);

	cuda_generate_gradients << <blocksPerGrid, threadsPerBlock >> > (prev_gradient_matrixes.data, prev_gradient_matrixes.pitch, prev_gradient_matrixes.cols_count, prev_gradient_matrixes.rows_count, gradients_device.cols_count, filter_size);
	cudacall(cudaGetLastError());

	cudaUnbindTexture(InputMatrixesRef);
}

void PoolingLayer::freeMemory() {
	
	cudaFree(gradients_device.data);
	cudaFree(outputs_devices.data);
}
