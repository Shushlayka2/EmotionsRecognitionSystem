#include "Tester.h"
#include "DigitImageLoadingService.h"

void Tester::test(Network& network) {

	int number_of_images;
	Tensor* testing_dataset = DigitImageLoadingService::read_mnist_images("t10k-images.idx3-ubyte", number_of_images);
	int* testing_labels = DigitImageLoadingService::read_mnist_labels("t10k-labels.idx1-ubyte", number_of_images);
	network.set_status(Status::Running);
	int correct_count = 0;
	
	for (int i = 0; i < number_of_images; i++)
	{
		network.set_inputs(testing_dataset[i]);
		network.run();
		int pred_res = network.get_result();
		int real_res = testing_labels[i];
		if (pred_res == real_res)
			correct_count++;
		network.calc_error(testing_labels[i]);
		network.free_inputs();
	}
	printf("\tNetork error: %f\n\tCorrect matches count: %d of %d", network.get_common_error(number_of_images), correct_count, number_of_images);
}