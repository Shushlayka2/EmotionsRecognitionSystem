#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "Random.h"
//#include "ImageHandler.h"

#include <malloc.h>
#include <iostream>
#include <stdio.h>

int main()
{
	/*float* arr;
	arr = (float*)std::malloc(10);
	set_normal_random(arr, 10);
	for (int i = 0; i < 10; i++)
	{
		printf("%f\n", arr[i]);
	}*/

	/*ImageHandler handler;
	int rows, cols;
	float** matrix = handler.convert(0, 0, rows, cols);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			printf("%f ", matrix[i][j]);
		}
		printf("\n");
	}
	printf("%i %i", rows, cols);*/

	return 0;
}