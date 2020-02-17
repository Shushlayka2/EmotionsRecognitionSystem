#include <fstream>

#include "Network.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define IMAGE_PITCH_OFFSET 896

Network::Network(ConfigHandler& configurationHandler, Status status) {
	
	params_storage.set_status(status);
	this->configurationHandler = configurationHandler;
	image_size = configurationHandler.Value("image_size");
	filter_size = configurationHandler.Value("filter_size");
	pooling_filters_size = configurationHandler.Value("pooling_filters_size");
	convolutional_layers_count = configurationHandler.Value("convolution_layers_count");
	fully_connected_layers_count = configurationHandler.Value("fully_connected_layers_count");
	convolutional_layers_filters_count = configurationHandler.VectorValue("convolutional_layers_filters_count");
	fully_connected_layers_neurons_count = configurationHandler.VectorValue("fully_connected_layers_neurons_count");
	init_layers();
}

void Network::run() {

	for (int i = 0; i < convolutional_layers_count; i++)
	{
		current_tensor = convolutionalLayers[i].forward(current_tensor);
		current_tensor = poolingLayers[i].forward(current_tensor, convolutionalLayers[i].gradients_device);
	}

	current_vector = fullyConnectedLayers[0].inputs_device;
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
		fullyConnectedLayers[i].backward(fullyConnectedLayers[i - 1].gradients_device);
	}

	fullyConnectedLayers[0].backward(intermediate_vector);
	Tensor cur_gradients_mb = poolingLayers[convolutional_layers_count - 1].gradients_device;
	cudaMemcpy2D(cur_gradients_mb.data, cur_gradients_mb.pitch, intermediate_vector, cur_gradients_mb.matrixes_size * sizeof(float),
		cur_gradients_mb.matrixes_size * sizeof(float), cur_gradients_mb.depth, cudaMemcpyDeviceToDevice);

	for (int i = convolutional_layers_count - 1; i > 0; i--)
	{
		poolingLayers[i].backward(convolutionalLayers[i].gradients_device);
		convolutionalLayers[i].correct();
		convolutionalLayers[i].backward(poolingLayers[i - 1].gradients_device);
	}
	poolingLayers[0].backward(convolutionalLayers[0].gradients_device);
	convolutionalLayers[0].correct();
}

void Network::init_layers() {

	int depth = 1;
	int current_size = image_size;
	for (int i = 0; i < convolutional_layers_count; i++)
	{
		int filters_count = convolutional_layers_filters_count[i];

		current_size = current_size - filter_size + 1;
		ConvolutionalLayer conv_layer = ConvolutionalLayer(filter_size, filters_count, depth, current_size, params_storage);

		depth = filters_count;

		current_size = current_size / 2 + (current_size % 2 == 0 ? 0 : 1);
		PoolingLayer pooling_layer = PoolingLayer(pooling_filters_size, current_size, depth);
		
		convolutionalLayers.push_back(conv_layer);
		poolingLayers.push_back(pooling_layer);
	}

	current_size *= current_size * depth;
	int first_fc_inputs_count = current_size;
	cudaMalloc((void**)&intermediate_vector, current_size * sizeof(float));

	Tensor& first_inputs = convolutionalLayers[0].inputs_device;
	first_inputs.cols_count = image_size; first_inputs.rows_count = image_size;
	first_inputs.matrixes_size = image_size * image_size; first_inputs.depth = 1;
	cudaMallocPitch((void**)&first_inputs.data, &first_inputs.pitch, first_inputs.matrixes_size * sizeof(float), first_inputs.depth);

	for (int i = 0; i < fully_connected_layers_count - 1; i++)
	{
		int next_layer_neurons_count = fully_connected_layers_neurons_count[i];
		FullyConnectedLayer fullyconnected_layer = FullyConnectedLayer(current_size, next_layer_neurons_count, params_storage, ActivationType::Sigmoid);
		fullyConnectedLayers.push_back(fullyconnected_layer);
		current_size = next_layer_neurons_count;
	}
	int next_layer_neurons_count = fully_connected_layers_neurons_count[fully_connected_layers_count - 1];
	FullyConnectedLayer fullyconnected_layer = FullyConnectedLayer(current_size, next_layer_neurons_count, params_storage, ActivationType::Softmax);
	fullyConnectedLayers.push_back(fullyconnected_layer);

	cudaMalloc((void**)&fullyConnectedLayers[0].inputs_device, first_fc_inputs_count * sizeof(float));
}

void Network::set_total_inputs(float* image_matrix, const int number_of_images) {

	current_tensor = convolutionalLayers[0].inputs_device;
	cudaMallocPitch((void**)&total_inputs, &current_tensor.pitch, current_tensor.matrixes_size * sizeof(float), number_of_images);
	cudaMemcpy2D(total_inputs, current_tensor.pitch, image_matrix, current_tensor.matrixes_size * sizeof(float), current_tensor.matrixes_size * sizeof(float), number_of_images, cudaMemcpyHostToDevice);
}

void Network::set_inputs(const int image_num) {

	current_tensor = convolutionalLayers[0].inputs_device;
	current_tensor.data = total_inputs + (image_num * IMAGE_PITCH_OFFSET);
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
	cudaFree(fullyConnectedLayers[0].inputs_device);
	cudaFree(intermediate_vector);
	cudaFree(total_inputs);
	convolutionalLayers.clear();
	poolingLayers.clear();
	fullyConnectedLayers.clear();
	params_storage.reset();
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