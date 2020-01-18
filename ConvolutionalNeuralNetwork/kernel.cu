#include "Trainer.h"
#include "CustomException.h"

#include <iostream>

//class Test {
//public:
//	int** arr;
//	Test() {
//		arr = new int*[1];
//		arr[0] = new int[1];
//		arr[0][0] = 1;
//	}
//};
//
//class Test2 {
//public:
//	Test test;
//	Test2() {
//		test = Test();
//	}
//};
//
//int main()
//{
//	Test2 test2 = Test2();
//	std::cout << test2.test.arr[0][0];
//}

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
	return 0;
}