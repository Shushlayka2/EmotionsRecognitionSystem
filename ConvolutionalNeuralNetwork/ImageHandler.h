#pragma once
#include <string>

#include "MatrixBlock.h"

class ImageHandler {
private:
	static std::string directory_path;

	std::string convert_id_to_file(const int id);

public:
	void convert(MatrixBlock& matrix_block, const int num, const int id);
};