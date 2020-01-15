#pragma once

#include <iostream>
#include <time.h>

#define AbsMaxVal 1000

class RNG
{
public:
	RNG()
	{
		srand(time(NULL));
	}

	virtual float next()
	{
		return (rand() / (float)RAND_MAX)* (AbsMaxVal * 2 + 1) + -1 * AbsMaxVal;
	}
};