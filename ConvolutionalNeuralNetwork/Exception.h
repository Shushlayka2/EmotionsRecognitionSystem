#pragma once

#include "cuda_runtime.h"
#include <cublas_v2.h>
#include <stdexcept>
#include <fstream>

using namespace std;

#define throw_line(arg) throw custom_exception(arg, __LINE__)
#define cublascall(call) {cublasStatus_t status = (call); if(CUBLAS_STATUS_SUCCESS != status) throw_line("" + status);}
#define cudacall(call) {cudaError_t err = (call); if(cudaSuccess != err) throw_line(cudaGetErrorString(err));}

class custom_exception : public runtime_error {
private:
	ofstream log_file;
public:
	custom_exception(const string& arg, int line) :
		runtime_error(arg), log_file("log.txt", ofstream::app) {
		log_file << line << ": " << arg << endl;
	}
	void destroy()
	{
		log_file.close();
	}
};