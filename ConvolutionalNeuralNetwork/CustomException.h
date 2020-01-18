#pragma once

#include <cublas_v2.h>
#include <stdexcept>
#include <fstream>

#include "cuda_runtime.h"

#define throw_line(arg) throw CustomException(arg, __FILE__,  __LINE__)
#define cublascall(call) {cublasStatus_t status = (call); if(CUBLAS_STATUS_SUCCESS != status) throw_line("" + status);}
#define cudacall(call) {cudaError_t err = (call); if(cudaSuccess != err) throw_line(cudaGetErrorString(err));}

class CustomException : public std::runtime_error {
private:
	static std::ofstream log_file;
public:
	CustomException(const std::string&, const std::string&, const int);

	void CloseLogFile();
};