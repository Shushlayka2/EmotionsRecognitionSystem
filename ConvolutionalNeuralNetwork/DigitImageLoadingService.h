#pragma once

#include <string>

#include "Tensor.h"

typedef unsigned char uchar;

static class DigitImageLoadingService {
public:
	static Tensor* read_mnist_images(std::string full_path, int& number_of_images);
	static int* read_mnist_labels(std::string full_path, int& number_of_labels);
};