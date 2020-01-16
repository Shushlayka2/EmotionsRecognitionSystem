#include "CustomException.h"

std::ofstream CustomException::log_file("log.txt", std::ofstream::app);

CustomException::CustomException(const std::string& arg, const int line) :
	runtime_error(arg) {
	log_file << line << ": " << arg << std::endl;
}

void CustomException::CloseLogFile() {
	log_file.close();
}