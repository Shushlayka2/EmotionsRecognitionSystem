#pragma once

#include <vector>

#include "Hub.h"
#include "Tensor.h"
#include "ConfigHandler.h"
#include "ConvolutionalLayer.h"
#include "PoolingLayer.h"
#include "FullyConnectedLayer.h"

class Network {
private:
	int filter_size;
	int image_size;
	int pooling_filters_size;
	int convolutional_layers_count;
	int fully_connected_layers_count;

	Hub params_storage;
	ConfigHandler configurationHandler;

	Tensor current_tensor;
	float* current_vector;
	float* intermediate_vector;
	float* total_inputs;
	std::vector<int> convolutional_layers_filters_count;
	std::vector<int> fully_connected_layers_neurons_count;
	std::vector<ConvolutionalLayer> convolutionalLayers;
	std::vector<PoolingLayer> poolingLayers;
	std::vector<FullyConnectedLayer> fullyConnectedLayers;

	void init_layers();

public:
	Network(ConfigHandler& configurationHandler, Status status);
	void run();
	void correct(const int correct_result);
	void set_total_inputs(float* image_matrix, const int number_of_images);
	void set_inputs(const int image_num);
	void free_memory();
	int get_result();
	void calc_error(int correct_result);
	float get_common_error(const int set_size);
	void set_status(Status status);
	void save_params();
};