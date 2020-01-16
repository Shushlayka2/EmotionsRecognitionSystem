#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "Random.h"
#include "ImageHandler.h"
#include "Matrix.h"

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

	ImageHandler handler;
	Matrix matrix = handler.convert(0, 0);
	for (int i = 0; i < matrix.rows_count; i++)
	{
		for (int j = 0; j < matrix.cols_count; j++)
		{
			printf("%f ", matrix.elements[i][j]);
		}
		printf("\n");
	}
	printf("%i %i", matrix.rows_count, matrix.cols_count);

	return 0;
}