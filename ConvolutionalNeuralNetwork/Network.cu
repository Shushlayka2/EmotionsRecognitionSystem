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
	
	Tensor& current_matrix_block = inputs_device;

	for (int i = 0; i < convolutional_layers_count; i++)
	{
		current_matrix_block = convolutionalLayers[i].forward(current_matrix_block);
		current_matrix_block = poolingLayers[i].forward(current_matrix_block, convolutionalLayers[i].filters_gr_device);
	}

	float* current_input_vector;
	cudaMalloc((void**)&current_input_vector, current_matrix_block.matrixes_size * current_matrix_block.depth * sizeof(float));
	cudaMemcpy2D(current_input_vector, current_matrix_block.matrixes_size * sizeof(float), current_matrix_block.data, current_matrix_block.pitch,
		current_matrix_block.matrixes_size * sizeof(float), current_matrix_block.depth, cudaMemcpyDeviceToDevice);
	for (int i = 0; i < fully_connected_layers_count; i++)
	{
		current_input_vector = fullyConnectedLayers[i].forward(current_input_vector);
	}
}

void Network::correct(int correct_result) {

	fullyConnectedLayers[fully_connected_layers_count - 1].set_gradients(correct_result);
	for (int i = fully_connected_layers_count - 1; i > 0; i--)
	{
		fullyConnectedLayers[i].backward(fullyConnectedLayers[i - 1].get_gradients());
		fullyConnectedLayers[i].correct();
	}

	Tensor cur_gradients_mb = poolingLayers[convolutional_layers_count - 1].gradients_device;
	float* first_pl_gr_vector_device;
	cudaMalloc((void**)&first_pl_gr_vector_device, cur_gradients_mb.matrixes_size * cur_gradients_mb.depth * sizeof(float));
	fullyConnectedLayers[0].backward(first_pl_gr_vector_device);

	cudaMemcpy2D(cur_gradients_mb.data, cur_gradients_mb.pitch, first_pl_gr_vector_device, cur_gradients_mb.matrixes_size * sizeof(float),
		cur_gradients_mb.matrixes_size * sizeof(float), cur_gradients_mb.depth, cudaMemcpyDeviceToDevice);

	for (int i = convolutional_layers_count - 1; i > 0; i--)
	{
		poolingLayers[i].backward(convolutionalLayers[i].filters_gr_device);
		convolutionalLayers[i].correct();
		convolutionalLayers[i].backward(poolingLayers[i - 1].gradients_device);
	}
	poolingLayers[0].backward(convolutionalLayers[0].filters_gr_device);
	convolutionalLayers[0].correct();
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
		prev_layer_neurons_count = next_layer_neurons_count;
	}
}

void Network::set_inputs(Tensor& image_matrix_block) {

	inputs_device = image_matrix_block;
	float* data_host = image_matrix_block.data;
	cudaMallocPitch((void**)&inputs_device.data, &inputs_device.pitch, inputs_device.matrixes_size * sizeof(float), inputs_device.depth);
	cudaMemcpy2D(inputs_device.data, inputs_device.pitch, data_host, inputs_device.matrixes_size * sizeof(float), inputs_device.matrixes_size * sizeof(float), inputs_device.depth, cudaMemcpyHostToDevice);
}

int Network::get_result() {
	return fullyConnectedLayers[fully_connected_layers_count - 1].get_result();
}

void Network::free_inputs() {

	cudaFree(inputs_device.data);
}