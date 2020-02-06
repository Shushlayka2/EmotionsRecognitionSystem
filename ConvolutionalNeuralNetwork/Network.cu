#include <fstream>

#include "Network.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

Network::Network(ConfigHandler& configurationHandler, Status status) {
	
	params_storage.set_status(status);
	if (status == Status::Training)
		params_storage.clear_params();
	this->configurationHandler = configurationHandler;
	image_size = configurationHandler.Value("image_size");
	filter_size = configurationHandler.Value("filter_size");
	amount_of_filters = configurationHandler.Value("amount_of_filters");
	pooling_filters_size = configurationHandler.Value("pooling_filters_size");
	convolutional_layers_count = configurationHandler.Value("convolution_layers_count");
	fully_connected_layers_count = configurationHandler.Value("fully_connected_layers_count");
	fully_connected_layers_neurons_count = configurationHandler.VectorValue("fully_connected_layers_neurons_count");
	init_layers();
}

void Network::run() {

	for (int i = 0; i < convolutional_layers_count; i++)
	{
		current_tensor = convolutionalLayers[i].forward(current_tensor);
		current_tensor = poolingLayers[i].forward(current_tensor, convolutionalLayers[i].get_gradients());
	}

	cudaMalloc((void**)&current_vector, current_tensor.matrixes_size * current_tensor.depth * sizeof(float));
	cudaMemcpy2D(current_vector, current_tensor.matrixes_size * sizeof(float), current_tensor.data, current_tensor.pitch,
		current_tensor.matrixes_size * sizeof(float), current_tensor.depth, cudaMemcpyDeviceToDevice);

	for (int i = 0; i < fully_connected_layers_count; i++)
	{
		current_vector = fullyConnectedLayers[i].forward(current_vector);
	}
}

void Network::correct(const int correct_result) {

	fullyConnectedLayers[fully_connected_layers_count - 1].set_gradients(correct_result);
	for (int i = fully_connected_layers_count - 1; i > 0; i--)
	{
		fullyConnectedLayers[i].backward(fullyConnectedLayers[i - 1].get_gradients());
	}

	Tensor cur_gradients_mb = poolingLayers[convolutional_layers_count - 1].get_gradients();
	float* first_pl_gr_vector_device;
	cudaMalloc((void**)&first_pl_gr_vector_device, cur_gradients_mb.matrixes_size * cur_gradients_mb.depth * sizeof(float));
	fullyConnectedLayers[0].backward(first_pl_gr_vector_device);

	cudaMemcpy2D(cur_gradients_mb.data, cur_gradients_mb.pitch, first_pl_gr_vector_device, cur_gradients_mb.matrixes_size * sizeof(float),
		cur_gradients_mb.matrixes_size * sizeof(float), cur_gradients_mb.depth, cudaMemcpyDeviceToDevice);

	for (int i = convolutional_layers_count - 1; i > 0; i--)
	{
		poolingLayers[i].backward(convolutionalLayers[i].get_gradients());
		convolutionalLayers[i].correct();
		convolutionalLayers[i].backward(poolingLayers[i - 1].get_gradients());
	}
	poolingLayers[0].backward(convolutionalLayers[0].get_gradients());
	convolutionalLayers[0].correct();
}

void Network::init_layers() {

	int depth = 1;
	int prev_layer_neurons_count = image_size;
	for (int i = 0; i < convolutional_layers_count; i++)
	{
		depth *= amount_of_filters;

		prev_layer_neurons_count = prev_layer_neurons_count - filter_size + 1;
		ConvolutionalLayer conv_layer = ConvolutionalLayer(filter_size, amount_of_filters, prev_layer_neurons_count, depth, params_storage);

		prev_layer_neurons_count = prev_layer_neurons_count / 2 + (prev_layer_neurons_count % 2 == 0 ? 0 : 1);
		PoolingLayer pooling_layer = PoolingLayer(pooling_filters_size, prev_layer_neurons_count, depth);
		
		convolutionalLayers.push_back(conv_layer);
		poolingLayers.push_back(pooling_layer);
	}

	prev_layer_neurons_count *= prev_layer_neurons_count * depth;

	for (int i = 0; i < fully_connected_layers_count - 1; i++)
	{
		int next_layer_neurons_count = fully_connected_layers_neurons_count[i];
		FullyConnectedLayer fullyconnected_layer = FullyConnectedLayer(prev_layer_neurons_count, next_layer_neurons_count, params_storage);
		fullyConnectedLayers.push_back(fullyconnected_layer);
		prev_layer_neurons_count = next_layer_neurons_count;
	}
	int next_layer_neurons_count = fully_connected_layers_neurons_count[fully_connected_layers_count - 1];
	FullyConnectedLayer fullyconnected_layer = FullyConnectedLayer(prev_layer_neurons_count, next_layer_neurons_count, params_storage, ActivationType::Softmax);
	fullyConnectedLayers.push_back(fullyconnected_layer);
}

void Network::set_inputs(Tensor& image_matrix_block) {

	convolutionalLayers[0].freeInputs();
	fullyConnectedLayers[0].freeInputs();
	current_tensor = image_matrix_block;
	float* data_host = image_matrix_block.data;
	cudaMallocPitch((void**)&current_tensor.data, &current_tensor.pitch, current_tensor.matrixes_size * sizeof(float), current_tensor.depth);
	cudaMemcpy2D(current_tensor.data, current_tensor.pitch, data_host, current_tensor.matrixes_size * sizeof(float), current_tensor.matrixes_size * sizeof(float), current_tensor.depth, cudaMemcpyHostToDevice);
}

int Network::get_result() {
	
	return fullyConnectedLayers[fully_connected_layers_count - 1].get_result();
}

void Network::calc_error(int correct_result) {

	fullyConnectedLayers[fully_connected_layers_count - 1].calc_error(correct_result);
}

float Network::get_common_error(const int set_size) {
	
	return fullyConnectedLayers[fully_connected_layers_count - 1].get_common_error(set_size);
}

void Network::free_memory() {

	for (int i = 0; i < convolutional_layers_count; i++)
	{
		convolutionalLayers[i].freeMemory();
		poolingLayers[i].freeMemory();
	}

	for (int i = 0; i < fully_connected_layers_count; i++)
		fullyConnectedLayers[i].freeMemory();

	convolutionalLayers.clear();
	poolingLayers.clear();
	fullyConnectedLayers.clear();
	params_storage.reset();
	cudaFree(current_tensor.data);
	cudaFree(current_vector);
}

void Network::set_status(Status status) {

	params_storage.set_status(status);
}

void Network::save_params() {

	for (int i = 0; i < convolutional_layers_count; i++)
		convolutionalLayers[i].save_params(params_storage);

	for (int i = 0; i < fully_connected_layers_count; i++)
		fullyConnectedLayers[i].save_params(params_storage);
}