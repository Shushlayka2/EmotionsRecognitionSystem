#include "Trainer.h"
#include "Network.h"
#include "ConfigHandler.h"
#include "ImageHandler.h"

void Trainer::train() {
	ConfigHandler configHandler("config.txt");
	ImageHandler imageHandler;
	Network network(configHandler);
	int train_chunk = configHandler.Value("train_chunk");
	for (int i = 0; i < 10; i++)
	{
		for (int j = 0; j < train_chunk; j++)
		{
			MatrixBlock input_image_matrix;
			imageHandler.convert(input_image_matrix, i, j);
			network.run(input_image_matrix);
		}
	}
}