#include "Tester.h"
#include "DigitImageLoadingService.h"

void Tester::test(Network& network) {

	int number_of_images;
	float* testing_dataset = DigitImageLoadingService::read_mnist_images("t10k-images.idx3-ubyte", number_of_images);
	int* testing_labels = DigitImageLoadingService::read_mnist_labels("t10k-labels.idx1-ubyte", number_of_images);
	network.set_status(Status::Running);
	int correct_count = 0;
	
	network.set_total_inputs(testing_dataset, number_of_images);
	for (int i = 0; i < number_of_images; i++)
	{
		network.set_inputs(i);
		network.run();
		int pred_res = network.get_result();
		int real_res = testing_labels[i];
		if (pred_res == real_res)
			correct_count++;
		network.calc_error(testing_labels[i]);
	}
	printf("\tNetork error: %f\n\tCorrect matches count: %d of %d\n", network.get_common_error(number_of_images), correct_count, number_of_images);
	delete[] testing_labels;
	delete[] testing_dataset;
}