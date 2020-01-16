#include <opencv2/opencv.hpp>

#include "ImageHandler.h"

std::string ImageHandler::directory_path = "..\\..\\TrainingImages\\numbers\\UNCATEGORIZED";

std::string ImageHandler::convert_id_to_file(const int id) {
	std::string id_str = std::to_string(id);
	int num_length = id_str.length();
	for (int i = num_length; i < 7; i++)
		id_str = "0" + id_str;
	id_str = "number-" + id_str;
	id_str += ".PNG";
	return id_str;
}

Matrix& ImageHandler::convert(const int num, const int id) {
	Matrix matrix;
	cv::Mat image = cv::imread(directory_path + "\\" + std::to_string(num) + "\\" + convert_id_to_file(id));
	matrix.rows_count = image.rows; matrix.cols_count = image.cols;
	matrix.elements = new float* [matrix.rows_count];
	for (int i = 0; i < matrix.rows_count; i++)
	{
		matrix.elements[i] = new float[matrix.cols_count];
		for (int j = 0; j < matrix.cols_count; j++)
		{
			matrix.elements[i][j] = (float)image.data[i * matrix.rows_count + j];
		}
	}
	return matrix;
}