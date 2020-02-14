#include <ctime>

#include "Trainer.h"
#include "DigitImageLoadingService.h"

void Trainer::train(Network& network, ConfigHandler configurationHandler) {
	
	int number_of_images;
	float* training_dataset = DigitImageLoadingService::read_mnist_images("train-images.idx3-ubyte", number_of_images);
	int* training_labels = DigitImageLoadingService::read_mnist_labels("train-labels.idx1-ubyte", number_of_images);

	int repetitions_count = configurationHandler.Value("repetitions_count");
	float epochs_count = configurationHandler.Value("epochs_count");
	int elapsedTime;

	network.set_total_inputs(training_dataset, number_of_images);
	for (int i = 0; i < epochs_count; i++)
	{
		//clock_t begin, end; 
		clock_t begin = clock();
		for (int j = 0; j < number_of_images; j++)
		{	
			//begin = clock();
			network.set_inputs(j);
			//end = clock();
			//printf("1 Elapsed time: %d\n", end - begin);
			//begin = clock();
			network.run();
			//end = clock();
			//printf("2 Elapsed time: %d\n", end - begin);
			for (int l = 0; l < repetitions_count; l++)
			{
				//begin = clock();
				network.correct(training_labels[j]);
				//end = clock();
				//printf("3 Elapsed time: %d\n", end - begin);
			}
		}
		clock_t end = clock();

		printf("%d epoch:\n\tElapsed time: %f\n", i, double(end - begin) / CLOCKS_PER_SEC);
	}

	save_params(network);
 	delete[] training_dataset;
	delete[] training_labels;
}

inline void Trainer::save_params(Network& network) {

	network.save_params();
}