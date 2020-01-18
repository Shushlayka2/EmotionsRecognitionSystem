#pragma once

#include <vector>

#include "MatrixBlock.h"
#include "ConfigHandler.h"

class Network {
private:
	std::vector<MatrixBlock> feature_maps;
	std::vector<MatrixBlock> pooled_feature_maps;
	ConfigHandler configurationHandler;

public:
	Network(ConfigHandler& configurationHandler);

	void run(MatrixBlock& image_matrix_block);
};