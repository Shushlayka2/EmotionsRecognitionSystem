#pragma once

#include "Tensor.h"
#include "Network.h"

class Trainer {
private:
	void save_params(Network& network);
public:
	void train(Network& network, ConfigHandler configurationHandler);
};