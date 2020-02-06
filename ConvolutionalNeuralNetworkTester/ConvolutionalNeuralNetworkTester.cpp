#include <fstream>

#include "pch.h"
#include "CppUnitTest.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Random.h"
#include "Network.h"
#include "ConfigHandler.h"
#include "CustomException.h"
#include "PoolingLayer.h"
#include "ConvolutionalLayer.h"
#include "FullyConnectedLayer.h"
#include "DigitImageLoadingService.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace ConvolutionalNeuralNetworkTester
{
	TEST_CLASS(ConvolutionalNeuralNetworkTester)
	{
	private:
		std::fstream file;
		Hub params_storage;
		Tensor inputs_device;
		Tensor custom_device;
		Tensor gradients_device;

		void host_to_device(Tensor& mb) {
			float* data_host = mb.data;
			cudaMallocPitch((void**)&mb.data, &mb.pitch, mb.matrixes_size * sizeof(float), mb.depth);
			cudaMemcpy2D(mb.data, mb.pitch, data_host, mb.matrixes_size * sizeof(float), mb.matrixes_size * sizeof(float), mb.depth, cudaMemcpyHostToDevice);
		}

		float* matrix_to_vector(Tensor& mb) {

			float* vector;
			cudaMalloc((void**)&vector, mb.matrixes_size * mb.depth * sizeof(float));
			cudaMemcpy2D(vector, mb.matrixes_size * sizeof(float), mb.data, mb.pitch,
				mb.matrixes_size * sizeof(float), mb.depth, cudaMemcpyDeviceToDevice);
			return vector;
		}

		Tensor& init_custom_inputs() {
			Tensor result = Tensor(5, 5, 1);
			result.data = new float[25];
			for (int i = 0; i < 25; i++)
				result.data[i] = i;
			host_to_device(result);
			return result;
		}
		
		Tensor& init_gradients() {
			Tensor result = Tensor(5, 5, 1);
			cudaMallocPitch((void**)&result.data, &result.pitch, result.matrixes_size * sizeof(float), result.depth);
			return result;
		}

	public:	
		ConvolutionalNeuralNetworkTester()
		{
			int number_of_images;
			file.open("C:\\Users\\Bulat\\source\\repos\\EmotionsRecognitionSystem\\ConvolutionalNeuralNetworkTester\\result.txt", std::ios::out);
			Tensor* training_dataset = DigitImageLoadingService::read_mnist_images("C:\\Users\\Bulat\\source\\repos\\EmotionsRecognitionSystem\\ConvolutionalNeuralNetwork\\train-images.idx3-ubyte", number_of_images);
			host_to_device(training_dataset[0]);
			inputs_device = training_dataset[0];
			custom_device = init_custom_inputs();
			gradients_device = init_gradients();
			for (int i = 1; i < number_of_images; i++)
				free(training_dataset[i].data);
		}

		~ConvolutionalNeuralNetworkTester()
		{
			file.close();
			cudaFree(inputs_device.data);
			cudaFree(custom_device.data);
			cudaFree(gradients_device.data);
		}

		TEST_METHOD(InputsTesting)
		{
			float* inputs_host;
			inputs_host = (float*)malloc(inputs_device.matrixes_size * inputs_device.depth * sizeof(float));
			cudaMemcpy2D(inputs_host, inputs_device.matrixes_size * sizeof(float), inputs_device.data, inputs_device.pitch, inputs_device.matrixes_size * sizeof(float), inputs_device.depth, cudaMemcpyDeviceToHost);
			
			for (int i = 0; i < inputs_device.depth; i++)
			{
				for (int j = 0; j < inputs_device.rows_count; j++)
				{
					for (int l = 0; l < inputs_device.cols_count; l++)
					{
						char buffer[100];
						sprintf(buffer, "%f ", inputs_host[i * inputs_device.matrixes_size + j * inputs_device.cols_count + l]);
						Assert::IsTrue(inputs_host[i * inputs_device.matrixes_size + j * inputs_device.cols_count + l] < 256.0f
							&& inputs_host[i * inputs_device.matrixes_size + j * inputs_device.cols_count + l] >= 0.0f);
						file << buffer;
					}
					file << std::endl;
				}
				file << std::endl;
			}
			file << std::endl;

			free(inputs_host);
		}

		TEST_METHOD(InputsToVectorTesting)
		{
			float* input_vector_device = matrix_to_vector(custom_device);
			int size = custom_device.matrixes_size * custom_device.depth;
			float* vector_host;
			vector_host = (float*)malloc(size * sizeof(float));
			cudaMemcpy(vector_host, input_vector_device, size * sizeof(float), cudaMemcpyDeviceToHost);
			for (int i = 0; i < size; i++)
			{
				char buffer[100];
				sprintf(buffer, "%f ", vector_host[i]);
				file << buffer;
			}
			file << std::endl;
		}

		TEST_METHOD(RandomTesting)
		{
			int rows = 10, cols = 9;
			float* result_host;
			float* result_device;
			size_t pitch;

			result_host = (float*)malloc(rows * cols * sizeof(float));
			result_device = set_normal_random(cols, rows, pitch);
			cudaMemcpy2D(result_host, cols * sizeof(float), result_device, pitch, cols * sizeof(float), rows, cudaMemcpyDeviceToHost);

			for (int i = 0; i < rows; i++)
			{
				for (int j = 0; j < cols; j++)
				{
					char buffer[100];
					sprintf(buffer, "%f ", result_host[i * cols + j]);
					Assert::IsTrue(abs(result_host[i * cols + j]) < 5.0f);
					file << buffer;
				}
				file << std::endl;
			}
			file << std::endl;

			cudaFree(result_device);
			free(result_host);
		}

		TEST_METHOD(ConvolveTesting)
		{
			int filter_size = 3, amount_of_filters = 3,
				gradient_size = custom_device.cols_count - filter_size + 1;

			ConvolutionalLayer conv_layer = ConvolutionalLayer(filter_size, amount_of_filters, gradient_size, filter_size, params_storage);

			custom_device = conv_layer.forward(custom_device);
			float* output_host;
			output_host = (float*)malloc(custom_device.matrixes_size * custom_device.depth * sizeof(float));
			cudaMemcpy2D(output_host, custom_device.matrixes_size * sizeof(float), custom_device.data, custom_device.pitch, custom_device.matrixes_size * sizeof(float), custom_device.depth, cudaMemcpyDeviceToHost);

			for (int i = 0; i < custom_device.depth; i++)
			{
				for (int j = 0; j < custom_device.rows_count; j++)
				{
					for (int l = 0; l < custom_device.cols_count; l++)
					{
						char buffer[100];
						sprintf(buffer, "%f ", output_host[i * custom_device.matrixes_size + j * custom_device.cols_count + l]);
						Assert::IsTrue(output_host[i * custom_device.matrixes_size + j * custom_device.cols_count + l] >= 0.0f);
						file << buffer;
					}
					file << std::endl;
				}
				file << std::endl;
			}
			file << std::endl;

			cudaFree(gradients_device.data);
			gradients_device = conv_layer.get_gradients();

			free(output_host);
		}

		TEST_METHOD(PoolingTesting)
		{
			int filter_size = 2, gradient_size = custom_device.cols_count / filter_size + (custom_device.cols_count % filter_size == 0 ? 0 : 1);
			PoolingLayer pooling_layer = PoolingLayer(filter_size, gradient_size, custom_device.depth);

			custom_device = pooling_layer.forward(custom_device, gradients_device);
			float* output_host = (float*)malloc(custom_device.matrixes_size * custom_device.depth * sizeof(float));
			cudaMemcpy2D(output_host, custom_device.matrixes_size * sizeof(float), custom_device.data, custom_device.pitch, custom_device.matrixes_size * sizeof(float), custom_device.depth, cudaMemcpyDeviceToHost);

			for (int i = 0; i < custom_device.depth; i++)
			{
				for (int j = 0; j < custom_device.rows_count; j++)
				{
					for (int l = 0; l < custom_device.cols_count; l++)
					{
						char buffer[100];
						sprintf(buffer, "%f ", output_host[i * custom_device.matrixes_size + j * custom_device.cols_count + l]);
						file << buffer;
					}
					file << std::endl;
				}
				file << std::endl;
			}
			file << std::endl;

			free(output_host);
			output_host = (float*)malloc(gradients_device.depth * gradients_device.matrixes_size * sizeof(float));
			cudaMemcpy2D(output_host, gradients_device.matrixes_size * sizeof(float), gradients_device.data, gradients_device.pitch, gradients_device.matrixes_size * sizeof(float), gradients_device.depth, cudaMemcpyDeviceToHost);
			
			for (int i = 0; i < gradients_device.depth; i++)
			{
				for (int j = 0; j < gradients_device.rows_count; j++)
				{
					for (int l = 0; l < gradients_device.cols_count; l++)
					{
						char buffer[100];
						sprintf(buffer, "%f ", output_host[i * gradients_device.matrixes_size + j * gradients_device.cols_count + l]);
						file << buffer;
					}
					file << std::endl;
				}
				file << std::endl;
			}
			file << std::endl;

			free(output_host);
		}

		TEST_METHOD(FullyConnectedTesting)
		{
			FullyConnectedLayer fullyConnected_layer = FullyConnectedLayer(custom_device.matrixes_size * custom_device.depth, 10, params_storage);
			float* custom_vector_device = matrix_to_vector(custom_device);
			custom_vector_device = fullyConnected_layer.forward(custom_vector_device);
			float* vector_host;
			vector_host = (float*)malloc(5 * sizeof(float));
			cudaMemcpy(vector_host, custom_vector_device, 10 * sizeof(float), cudaMemcpyDeviceToHost);
			for (int i = 0; i < 10; i++)
			{
				char buffer[100];
				sprintf(buffer, "%f ", vector_host[i]);
				file << buffer;
			}
			file << std::endl;
		}

		TEST_METHOD(ForwardTesting)
		{
			custom_device = inputs_device;
			ConvolveTesting();
			PoolingTesting();
			InputsToVectorTesting();
			FullyConnectedTesting();
			custom_device = init_custom_inputs();
		}
	};
}
