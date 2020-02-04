#include "Trainer.h"
#include "Network.h"
#include "ConfigHandler.h"
#include "DigitImageLoadingService.h"

void Trainer::train() {
	
	//preproccess
	int number_of_images;
	ConfigHandler configHandler("config.txt");
	Network network(configHandler);
	training_dataset = DigitImageLoadingService::read_mnist_images("train-images.idx3-ubyte", number_of_images);
	training_labels = DigitImageLoadingService::read_mnist_labels("train-labels.idx1-ubyte", number_of_images);

	//training
	for (int i = 0; i < number_of_images / 10; i++)
	{
		network.set_inputs(training_dataset[i]);
		network.run();

		for (int j = 0; j < 2; j++)
			network.correct(training_labels[i]);
	}

	//test
	for (int i = 0; i < 10; i++)
	{
		network.set_inputs(training_dataset[i]);
		network.run();
		int result = network.get_result();
		int correct_result = training_labels[i];
		printf("Is correct: %d\n", correct_result == result);
	}
}