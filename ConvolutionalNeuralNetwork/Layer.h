#pragma once

#include <vector>

#include "RNG.h"

class Layer {
protected:
	const int in_size;
	const int out_size;

public:
	Layer(const int in_size, const int out_size) :
		in_size(in_size),
		out_size(out_size) {}
	
	virtual ~Layer();

	int get_in_size() const { return in_size; }
	int get_out_size() const { return out_size; }

	virtual void init(const float mu, const float sigma) = 0;
	virtual void forward(const float* prev_layer_output) = 0;

	virtual const float* output() const = 0;

	virtual void backprop(const float* prev_layer_output, const float* next_layer_data) = 0;
	virtual const float* backprop_data() const = 0;

	virtual std::vector<float> get_parameter() const = 0;
	virtual void set_parameters(const std::vector<float> param) = 0;
	virtual std::vector<float> get_derivatives() const = 0;
};