#pragma once
#include "MatrixBlock.h"

class Trainer {
public:
	MatrixBlock* training_dataset = nullptr;
	void train();
};