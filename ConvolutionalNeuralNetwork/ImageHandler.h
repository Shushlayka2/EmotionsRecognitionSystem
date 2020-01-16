#pragma once
#include <opencv2/opencv.hpp>
#include <string>

using namespace std;
using namespace cv;

class ImageHandler
{
private:
	static string directory_path;

	string convert_id_to_file(const int id)
	{
		string id_str = to_string(id);
		int num_length = id_str.length();
		for (int i = num_length; i < 7; i++)
			id_str = "0" + id_str;
		id_str = "number-" + id_str;
		id_str += ".PNG";
		return id_str;
	}

public:
	float** convert(const int num, const int id, int& rows, int& cols)
	{
		Mat image = imread(directory_path + "\\" + to_string(num) + "\\" + convert_id_to_file(id));
		rows = image.rows; cols = image.cols;
		float** matrix = new float*[rows];
		for (int i = 0; i < rows; i++)
		{
			matrix[i] = new float[cols];
			for (int j = 0; j < cols; j++)
			{
				matrix[i][j] = (float)image.data[i * rows + j];
			}
		}
		return matrix;
	}
};

string ImageHandler::directory_path = "..\\..\\TrainingImages\\numbers\\UNCATEGORIZED";