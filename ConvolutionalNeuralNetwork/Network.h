#pragma once

#include <vector>

#include "MatrixBlock.h"
#include "ConfigHandler.h"
#include "ConvolutionalLayer.h"
#include "FullyConnectedLayer.h"

class Network {
private:
	ConfigHandler configurationHandler;
	int convolutional_layers_count;
	int fully_connected_layers_count;
	std::vector<int> fully_connected_layers_neurons_count;

public:
	std::vector<ConvolutionalLayer> convolutionalLayers;
	std::vector<FullyConnectedLayer> fullyConnectedLayers;

	Network(ConfigHandler& configurationHandler);

	void run(MatrixBlock& image_matrix_block);
};