#include "Hub.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>

Hub::Hub() : inp_data_stream("params.txt", std::fstream::app), out_data_stream("params.txt", std::fstream::app), status(Status::Training) { }

Hub::~Hub() {

    inp_data_stream.close();
    out_data_stream.close();
}

void Hub::set_status(Status status) {

    this->status = status;
}

Status Hub::get_status() {
    
    return status;
}

void Hub::get_params(Tensor& tensor) {
    
    cudaMallocPitch((void**)&tensor.data, &tensor.pitch, tensor.matrixes_size * sizeof(float), tensor.depth);
    int arr_size = tensor.matrixes_size * tensor.depth;
    float* buffer = new float[arr_size];
    for (int i = 0; i < arr_size; i++)
        inp_data_stream >> buffer[i];
    cudaMemcpy2D(tensor.data, tensor.pitch, buffer, tensor.matrixes_size * sizeof(float),
        tensor.matrixes_size * sizeof(float), tensor.depth, cudaMemcpyHostToDevice);
    delete[] buffer;
}

float* Hub::get_params(const int arr_size) {

    float* arr;
    cudaMalloc((void**)&arr, arr_size * sizeof(float));
    float* buffer = new float[arr_size];
    for (int i = 0; i < arr_size; i++)
        inp_data_stream >> buffer[i];
    cudaMemcpy(arr, buffer, arr_size * sizeof(float), cudaMemcpyHostToDevice);
    delete[] buffer;

    return arr;
}

void Hub::set_params(Tensor& tensor) {

    int arr_size = tensor.matrixes_size * tensor.depth;
    float* buffer = new float[arr_size];
    cudaMemcpy2D(buffer, tensor.matrixes_size * sizeof(float), tensor.data, tensor.pitch,
        tensor.matrixes_size * sizeof(float), tensor.depth, cudaMemcpyDeviceToHost);
    for (int i = 0; i < arr_size; i++)
        out_data_stream << buffer[i] << " ";
    out_data_stream << std::endl;
    
    delete[] buffer;
}

void Hub::set_params(float* arr, const int arr_size) {

    float* buffer = new float[arr_size];
    cudaMemcpy(buffer, arr, arr_size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < arr_size; i++)
        out_data_stream << buffer[i] << " ";
    out_data_stream << std::endl;
    delete[] buffer;
}

void Hub::clear_params() {

    std::ofstream cleaner("params.txt", std::ofstream::out | std::ofstream::trunc);
    cleaner.close();
}

void Hub::reset() {
    inp_data_stream.clear();
    inp_data_stream.seekg(0, std::ios::beg);
}