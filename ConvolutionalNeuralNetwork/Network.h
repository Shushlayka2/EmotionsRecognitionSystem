#pragma once

#include <vector>

#include "MatrixBlock.h"
#include "ConfigHandler.h"
#include "ConvolutionalLayer.h"

class Network {
private:
	ConfigHandler configurationHandler;
	int convolutional_layers_count;

public:
	std::vector<ConvolutionalLayer> convolutionalLayers;

	Network(ConfigHandler& configurationHandler);

	void run(MatrixBlock& image_matrix_block);
};