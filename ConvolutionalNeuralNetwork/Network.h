#pragma once

#include <vector>

#include "MatrixBlock.h"
#include "ConfigHandler.h"
#include "ConvolutionalLayer.h"
#include "PoolingLayer.h"
#include "FullyConnectedLayer.h"

class Network {
private:
	int filter_size;
	int image_size;
	int amount_of_filters;
	int pooling_filters_size;
	int convolutional_layers_count;
	int fully_connected_layers_count;

	ConfigHandler configurationHandler;

	MatrixBlock inputs_device;
	std::vector<int> fully_connected_layers_neurons_count;
	std::vector<ConvolutionalLayer> convolutionalLayers;
	std::vector<PoolingLayer> poolingLayers;
	std::vector<FullyConnectedLayer> fullyConnectedLayers;

	void init_layers();

public:
	Network(ConfigHandler& configurationHandler);
	void run();
	void correct(int correct_result);
	void set_inputs(MatrixBlock& image_matrix_block);
	void free_inputs();
};