#pragma once
#include <string>

#include "Matrix.h"

class ImageHandler {
private:
	static std::string directory_path;

	std::string convert_id_to_file(const int id);

public:
	Matrix& convert(const int num, const int id);
};