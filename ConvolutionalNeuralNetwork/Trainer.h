#pragma once
#include "MatrixBlock.h"

class Trainer {
public:
	MatrixBlock* training_dataset = nullptr;
	int* training_labels = nullptr;
	void train();
};