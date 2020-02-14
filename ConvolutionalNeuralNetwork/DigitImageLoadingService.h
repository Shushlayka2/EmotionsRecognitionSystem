#pragma once

#include <string>

#include "Tensor.h"

typedef unsigned char uchar;

static class DigitImageLoadingService {
public:
	static float* read_mnist_images(char* full_path, int& number_of_images);
	static int* read_mnist_labels(char* full_path, int& number_of_labels);
};