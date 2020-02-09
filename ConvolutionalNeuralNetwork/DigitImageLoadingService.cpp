#include <fstream>

#include "DigitImageLoadingService.h"

#define BEGIN_OF_IMAGES 16
#define BEGIN_OF_LABELS 8
#define BEGIN_OF_MN 0
#define BEGIN_OF_SIZE 4
#define IMAGE_SIZE 784

int getVal(const unsigned char *buffer, int pos) {

    return (int)((buffer[pos] << 24) + (buffer[pos + 1] << 16) + (buffer[pos + 2] << 8) + buffer[pos + 3]);
}

float** DigitImageLoadingService::read_mnist_images(char* full_path, int& number_of_images) {

    FILE* file;
    long length;

    file = fopen(full_path, "rb");
    fseek(file, 0, SEEK_END);
    length = ftell(file);
    rewind(file);

    unsigned char* bufferImages = (unsigned char*)malloc((length + 1) * sizeof(unsigned char));
    fread(bufferImages, length, 1, file);
    fclose(file);
    
    int magic_number = getVal(bufferImages, BEGIN_OF_MN);
    if (magic_number != 2051) throw std::runtime_error("Invalid MNIST image file!");
    number_of_images = getVal(bufferImages, BEGIN_OF_SIZE);

    float** matrixes_dataset = new float*[number_of_images];
    *matrixes_dataset = new float[number_of_images * IMAGE_SIZE];
    for (int i = 1; i < number_of_images; i++) matrixes_dataset[i] = matrixes_dataset[i - 1] + IMAGE_SIZE;
    for (int i = 0; i < number_of_images; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            matrixes_dataset[i][j] = ((float)bufferImages[BEGIN_OF_IMAGES + i * IMAGE_SIZE + j]) / 255;
        }
    }
    free(bufferImages);
    return matrixes_dataset;
}

int* DigitImageLoadingService::read_mnist_labels(char* full_path, int& number_of_labels) {

    FILE* file;
    long length;

    file = fopen(full_path, "rb");
    fseek(file, 0, SEEK_END);
    length = ftell(file);
    rewind(file);

    unsigned char* bufferLabels = (unsigned char*)malloc((length + 1) * sizeof(unsigned char));
    fread(bufferLabels, length, 1, file);
    fclose(file);

    int magic_number = getVal(bufferLabels, BEGIN_OF_MN);
    if (magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");
    number_of_labels = getVal(bufferLabels, BEGIN_OF_SIZE);

    int* _dataset = new int[number_of_labels];
    for (int i = 0; i < number_of_labels; i++) {
        _dataset[i] = (int)bufferLabels[BEGIN_OF_LABELS + i];
    }
    free(bufferLabels);
    return _dataset;
}