#include <ctime>

#include "Trainer.h"
#include "DigitImageLoadingService.h"

void Trainer::train(Network& network, ConfigHandler configurationHandler) {
	
	//test
	/*int correct_count = 0;
	int number_of_test_images;
	float* testing_dataset = DigitImageLoadingService::read_mnist_images("t10k-images.idx3-ubyte", number_of_test_images);
	int* testing_labels = DigitImageLoadingService::read_mnist_labels("t10k-labels.idx1-ubyte", number_of_test_images);*/

	int number_of_images;
	float* training_dataset = DigitImageLoadingService::read_mnist_images("train-images.idx3-ubyte", number_of_images);
	int* training_labels = DigitImageLoadingService::read_mnist_labels("train-labels.idx1-ubyte", number_of_images);

	int repetitions_count = configurationHandler.Value("repetitions_count");
	float epochs_count = configurationHandler.Value("epochs_count");

	network.set_total_inputs(training_dataset, number_of_images);
	for (int i = 0; i < epochs_count; i++)
	{
		clock_t begin = clock();
		for (int j = 0; j < number_of_images; j++)
		{
			network.set_inputs(j);
			network.run();
			for (int l = 0; l < repetitions_count; l++)
				network.correct(training_labels[j]);
		}	
		clock_t end = clock();
		printf("%d epoch:\n\tElapsed time: %f\n", i, double(end - begin) / CLOCKS_PER_SEC);

		//test
		/*for (int j = 0; j < 1000; j++)
		{
			network.set_inputs(j);
			network.run();
			int pred_res = network.get_result();
			int real_res = training_labels[j];
			if (pred_res == real_res)
				correct_count++;
			network.calc_error(training_labels[j]);
		}
		printf("\tNetork error: %f\n\tCorrect matches count: %d of %d\n", network.get_common_error(1000), correct_count, 1000);
		correct_count = 0.0f;*/
	}

	save_params(network);
 	delete[] training_dataset;
	delete[] training_labels;
}

inline void Trainer::save_params(Network& network) {

	network.save_params();
}