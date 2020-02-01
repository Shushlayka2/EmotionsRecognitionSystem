#pragma once
#include "Tensor.h"

class Trainer {
public:
	Tensor* training_dataset = nullptr;
	int* training_labels = nullptr;
	void train();
};