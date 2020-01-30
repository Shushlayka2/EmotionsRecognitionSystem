#include "Network.h"
#include "CustomException.h"
#include "PoolingLayer.h"
#include "FullyConnectedLayer.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

Network::Network(ConfigHandler& configurationHandler) {
	this->configurationHandler = configurationHandler;
	this->image_size = configurationHandler.Value("image_size");
	this->filter_size = configurationHandler.Value("filter_size");
	this->amount_of_filters = configurationHandler.Value("amount_of_filters");
	this->pooling_filters_size = configurationHandler.Value("pooling_filters_size");
	this->convolutional_layers_count = configurationHandler.Value("convolution_layers_count");
	this->fully_connected_layers_count = configurationHandler.Value("fully_connected_layers_count");
	this->fully_connected_layers_neurons_count = configurationHandler.VectorValue("fully_connected_layers_neurons_count");
	init_layers();
}

void Network::run() {
	
	MatrixBlock& current_matrix_block = inputs_device;

	for (int i = 0; i < convolutional_layers_count; i++)
	{
		ConvolutionalLayer conv_layer = convolutionalLayers[i];
		current_matrix_block = conv_layer.forward(current_matrix_block);

		PoolingLayer pooling_layer = poolingLayers[i];
		current_matrix_block = pooling_layer.forward(current_matrix_block, conv_layer.gradients_device);
	}

	float* current_input_vector;
	cudaMalloc((void**)&current_input_vector, current_matrix_block.matrixes_size * current_matrix_block.depth * sizeof(float));
	cudaMemcpy2D(current_input_vector, current_matrix_block.matrixes_size * sizeof(float), current_matrix_block.data, current_matrix_block.pitch,
		current_matrix_block.matrixes_size * sizeof(float), current_matrix_block.depth, cudaMemcpyDeviceToDevice);
	for (int i = 0; i < fully_connected_layers_count; i++)
	{
		FullyConnectedLayer fullyconnected_layer = fullyConnectedLayers[i];
		current_input_vector = fullyconnected_layer.forward(current_input_vector);
	}
}

void Network::correct(int correct_result) {

	float* current_gradients = fullyConnectedLayers[fully_connected_layers_count - 1].set_gradients(correct_result);
	for (int i = fully_connected_layers_count - 1; i > 0; i--)
	{
		current_gradients = fullyConnectedLayers[i].backward(fullyConnectedLayers[i - 1].get_gradients());
	}

	MatrixBlock cur_gradients_mb = poolingLayers[convolutional_layers_count - 1].gradients_device;
	cudaMemcpy2D(cur_gradients_mb.data, cur_gradients_mb.pitch, current_gradients, cur_gradients_mb.matrixes_size * sizeof(float),
		cur_gradients_mb.matrixes_size * sizeof(float), cur_gradients_mb.depth, cudaMemcpyDeviceToDevice);


	for (int i = convolutional_layers_count - 1; i > 0; i--)
	{
		poolingLayers[i].backward(convolutionalLayers[i].gradients_device);
		convolutionalLayers[i].backward(poolingLayers[i - 1].gradients_device);
	}
	poolingLayers[0].backward(convolutionalLayers[0].gradients_device);
}

void Network::init_layers() {

	int depth = 1;
	int prev_layer_neurons_count = image_size;
	for (int i = 0; i < convolutional_layers_count; i++)
	{
		depth *= amount_of_filters;

		prev_layer_neurons_count = prev_layer_neurons_count - filter_size + 1;
		ConvolutionalLayer conv_layer = ConvolutionalLayer(filter_size, amount_of_filters, prev_layer_neurons_count, depth);

		prev_layer_neurons_count = prev_layer_neurons_count / 2 + (prev_layer_neurons_count % 2 == 0 ? 0 : 1);
		PoolingLayer pooling_layer = PoolingLayer(pooling_filters_size, prev_layer_neurons_count, depth);
		
		convolutionalLayers.push_back(conv_layer);
		poolingLayers.push_back(pooling_layer);
	}

	prev_layer_neurons_count *= prev_layer_neurons_count * depth;

	for (int i = 0; i < fully_connected_layers_count; i++)
	{
		int next_layer_neurons_count = fully_connected_layers_neurons_count[i];
		FullyConnectedLayer fullyconnected_layer = FullyConnectedLayer(prev_layer_neurons_count, next_layer_neurons_count);
		fullyConnectedLayers.push_back(fullyconnected_layer);
	}
}

void Network::set_inputs(MatrixBlock& image_matrix_block) {

	inputs_device = image_matrix_block;
	float* data_host = image_matrix_block.data;
	cudaMallocPitch((void**)&inputs_device.data, &inputs_device.pitch, inputs_device.matrixes_size * sizeof(float), inputs_device.depth);
	cudaMemcpy2D(inputs_device.data, inputs_device.pitch, data_host, inputs_device.matrixes_size * sizeof(float), inputs_device.matrixes_size * sizeof(float), inputs_device.depth, cudaMemcpyHostToDevice);
	free(data_host);
}

void Network::free_inputs() {

	cudaFree(inputs_device.data);
}