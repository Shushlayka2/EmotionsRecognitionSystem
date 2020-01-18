//#include "Layer.h"
//#include "Random.h"
//#include <malloc.h>
//
//template<typename Activation>
//class FullyConnected :public Layer {
//private:
//	float* weights_matrix;
//	float* biases_vector;
//	float* weights_derivatives_matrix;
//	float* biases_derivatives_matrix;
//	float* multiplication_matrix;
//	float* output_matrix;
//	float* inputs_derivatives_matrix;
//
//public:
//	FullyConnected(const int in_size, const int out_size) :
//		Layer()
//	{}
//
//	void init(const float mu, const float sigma) {
//		weights_matrix = (float*)malloc(in_size * out_size);
//		biases_vector = (float*)malloc(out_size);
//		weights_derivatives_matrix = (float*)malloc(in_size * out_size);
//		biases_derivatives_matrix = (float*)malloc(out_size);
//
//		set_normal_random(weights_matrix, in_size * out_size, mu, sigma);
//		set_normal_random(biases_vector, out_size, mu, sigma);
//	}
//
//	void forward(const float* prev_layer_data) {
//		
//	}
//};