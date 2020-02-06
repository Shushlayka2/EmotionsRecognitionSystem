#pragma once

#include <string>

#include "Tensor.h"

class ImageHandler {
private:
	static std::string directory_path;

	std::string convert_id_to_file(const int id);

public:
	void convert(Tensor& matrix_block, const int num, const int id);
};