#include "Trainer.h"
#include "CustomException.h"

#include <iostream>

int main()
{
	try
	{
		Trainer trainer;
		trainer.train();
	}
	catch (CustomException ex)
	{
		printf("Exception appeared! See log file!");
	}

	CustomException::CloseLogFile();
	return 0;
}