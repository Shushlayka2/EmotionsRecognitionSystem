#include <iostream>
#include <string>

#include "Tester.h"
#include "Trainer.h"
#include "Network.h"
#include "ConfigHandler.h"
#include "CustomException.h"

static ConfigHandler configHandler("config.txt");

int main()
{
	try
	{
		while (true)
		{
			std::cout << "Enter 'train' to train network / Enter 'test' to test network / Enter 'exit' for exit: ";
			std::string answer;
			std::cin >> answer;
			if (answer == "train")
			{
				Network network(configHandler, Status::Training);
				Trainer trainer;
				printf("Training started!\n");
				trainer.train(network, configHandler);
				std::cout << "Training completed!\n";
				network.free_memory();
			}
			else if (answer == "test")
			{
				Network network(configHandler, Status::Running);
				Tester tester;
				printf("Testing started!\n");
				tester.test(network);
				std::cout << "Testing completed!\n";
				network.free_memory();
			}
			else if (answer == "exit")
				break;
			else
				std::cout << "Undefined answer! Try again!\n";
		}
	}
	catch (CustomException ex)
	{
		printf("Exception appeared! See log file!");
	}

	CustomException::CloseLogFile();
	return 0;
}