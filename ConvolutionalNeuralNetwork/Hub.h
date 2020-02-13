#pragma once

#include <fstream>

#include "Tensor.h"

enum class Status {
	Training,
	Running,
	Undefined
};

class Hub {
private:
	std::ifstream inp_data_stream;
	std::ofstream out_data_stream;
	Status status;

	void clear_params();
public:
	Hub();
	~Hub();
	void set_status(Status status);
	Status get_status();
	void reset();
	float* get_params(const int arr_size);
	void get_params(Tensor& tensor);
	void set_params(Tensor& tensor);
	void set_params(float* arr, const int arr_size);
};