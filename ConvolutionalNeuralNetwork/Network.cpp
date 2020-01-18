#include "Network.h"
#include "ConvolutionalLayer.h"
#include "CustomException.h"

Network::Network(ConfigHandler& configurationHandler) {
	this->configurationHandler = configurationHandler;
}

void Network::run(MatrixBlock& image_matrix_block) {
	int filter_size = configurationHandler.Value("filter_size");
	int amount_of_filters = configurationHandler.Value("amount_of_filters");
	if (amount_of_filters > 10)
		throw_line("Amout of filters too large! Please, correct configuration!");
	MatrixBlock current_matrix_block = image_matrix_block;
	while (current_matrix_block.cols_count > filter_size && current_matrix_block.rows_count > filter_size)
	{
		//test
		/*filter_size = 3;
		amount_of_filters = 2;
		current_matrix_block = MatrixBlock(5, 5, 1);
		current_matrix_block.data = new float[25];
		for (int i = 0; i < 25; i++)
			current_matrix_block.data[i] = 1;*/

		ConvolutionalLayer conv_layer = ConvolutionalLayer(current_matrix_block, filter_size, amount_of_filters);
		conv_layer.forward();

		//test
		/*MatrixBlock feature_map = conv_layer.get_feature_map();
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
}