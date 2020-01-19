#pragma once

class FullyConnectedLayer {
private:
	int in_size;
	int out_size;
	float* weights_matrix;
	float* biases_vector;

	void activate_softmax(float* outputs_device, float* outputs);
	float* sum_particles_host(float* d_A_odd, int in_size, int out_size, int rows);
public:
	FullyConnectedLayer(int in_size, int out_size);

	float* forward(float* prev_layer_data);
};