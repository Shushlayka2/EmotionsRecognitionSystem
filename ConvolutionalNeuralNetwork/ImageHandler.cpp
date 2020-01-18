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

void ImageHandler::convert(MatrixBlock& matrix_block, const int num, const int id) {
	cv::Mat image = cv::imread(directory_path + "\\" + std::to_string(num) + "\\" + convert_id_to_file(id));
	matrix_block.cols_count = image.cols; matrix_block.rows_count = image.rows;
	matrix_block.matrixes_size = image.cols * image.rows; matrix_block.depth = image.dims - 1;
	matrix_block.matrixes = new float* [matrix_block.depth];

	for (int i = 0; i < matrix_block.depth; i++)
	{
		matrix_block.matrixes[i] = new float[matrix_block.matrixes_size];
		for (int j = 0; j < matrix_block.rows_count; j++)
		{
			for (int k = 0; k < matrix_block.cols_count; k++)
			{
				matrix_block.matrixes[i][j * image.cols + k] = (float)image.data[i * image.rows * image.cols + j * image.cols + k];
			}
		}
	}
}