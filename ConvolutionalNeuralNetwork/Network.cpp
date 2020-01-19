#include "Network.h"
#include "CustomException.h"
#include "PoolingLayer.h"
#include "FullyConnectedLayer.h"

Network::Network(ConfigHandler& configurationHandler) {
	this->configurationHandler = configurationHandler;
	this->convolutional_layers_count = configurationHandler.Value("convolution_layers_count");
	this->fully_connected_layers_count = configurationHandler.Value("fully_connected_layers_count");
	this->fully_connected_layers_neurons_count = configurationHandler.VectorValue("fully_connected_layers_neurons_count");
}

void Network::run(MatrixBlock& image_matrix_block) {
	int filter_size = configurationHandler.Value("filter_size");
	int amount_of_filters = configurationHandler.Value("amount_of_filters");
	int pooling_filters_size = configurationHandler.Value("pooling_filters_size");
	if (amount_of_filters > 10)
		throw_line("Amout of filters too large! Please, correct configuration!");
	MatrixBlock current_matrix_block = image_matrix_block;
	if (convolutionalLayers.size() == 0)
	{
		//First time run
		for (int i = 0; i < convolutional_layers_count; i++)
		{
			//test
			/*filter_size = 3;
			amount_of_filters = 2;
			current_matrix_block = MatrixBlock(5, 5, 1);
			current_matrix_block.data = new float[25];
			for (int i = 0; i < 25; i++)
				current_matrix_block.data[i] = i;*/

			ConvolutionalLayer conv_layer = ConvolutionalLayer(filter_size, amount_of_filters);
			current_matrix_block = conv_layer.forward(current_matrix_block);

			PooingLayer pooling_layer = PooingLayer(pooling_filters_size);
			current_matrix_block = pooling_layer.forward(current_matrix_block);
			convolutionalLayers.push_back(conv_layer);

			//test
			/*MatrixBlock feature_map = current_matrix_block;
			for (int k = 0; k < feature_map.depth; k++)
			{
				for (int l = 0; l < feature_map.rows_count; l++)
				{
					for (int m = 0; m < feature_map.cols_count; m++)
					{
						printf("%f ", feature_map.data[k * feature_map.matrixes_size + l * feature_map.cols_count + m]);
					}
					printf("\n");
				}
				printf("\n");
			}*/
		}

		for (int i = 0; i < fully_connected_layers_count; i++)
		{
			//test
			/*current_matrix_block.matrixes_size = 5;
			current_matrix_block.depth = 1;
			current_matrix_block.data = new float[5];
			for (int i = 0; i < 5; i++)
				current_matrix_block.data[i] = i;*/

			int next_layer_neurons_count = fully_connected_layers_neurons_count[i];
			FullyConnectedLayer fullyconnected_layer = FullyConnectedLayer(current_matrix_block.depth * current_matrix_block.matrixes_size, next_layer_neurons_count);
			fullyconnected_layer.forward(current_matrix_block.data);
		}
	}
}