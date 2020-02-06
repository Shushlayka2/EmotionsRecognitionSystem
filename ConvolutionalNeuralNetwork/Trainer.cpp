#include <ctime>

#include "Trainer.h"
#include "DigitImageLoadingService.h"

void Trainer::train(Network& network, ConfigHandler configurationHandler) {
	
	int number_of_images;
	Tensor* training_dataset = DigitImageLoadingService::read_mnist_images("train-images.idx3-ubyte", number_of_images);
	int* training_labels = DigitImageLoadingService::read_mnist_labels("train-labels.idx1-ubyte", number_of_images);

	int repetitions_count = configurationHandler.Value("repetitions_count");
	float epochs_count = configurationHandler.Value("epochs_count");
	int elapsedTime;

	for (int i = 0; i < epochs_count; i++)
	{
		clock_t begin = clock();
		for (int j = 0; j < number_of_images; j++)
		{
			network.set_inputs(training_dataset[j]);
			network.run();

			for (int l = 0; l < repetitions_count; l++)
				network.correct(training_labels[j]);
		}
		clock_t end = clock();
		
		printf("%d epoch:\n\tElapsed time: %f\n", i, double(end - begin) / CLOCKS_PER_SEC);
	}

	for (int i = 0; i < number_of_images; i++)
		delete training_dataset[i].data;

	save_params(network);
	delete training_dataset->data;
	delete training_labels;
}

inline void Trainer::save_params(Network& network) {

	network.save_params();
}