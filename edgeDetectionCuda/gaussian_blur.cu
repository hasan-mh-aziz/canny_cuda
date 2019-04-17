
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<opencv2/opencv.hpp>
#include<iostream>
#include <stdarg.h>
#include <cmath>
#include "gaussHeader.h"

using namespace std;
using namespace cv;

void ShowManyImages(string title, int nArgs, ...);

__global__ void gauss_gpu_kernel_v2(unsigned char *image, float* kernel, int kernel_size, int rows, int cols, float *blurred_image) {
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx > rows || idy > cols) {
		return;
	}

	int target_index = idx * cols + idy;

	int x, y, index, k_index;

	float sum = 0;
	int kx = threadIdx.z;
	x = idx - kernel_size / 2 + kx;
	if (x < 0) return;
	for (int ky = 0; ky < kernel_size; ky++) {
		y = idy - kernel_size / 2 + ky;
		if (y < 0) continue;

		index = x * cols + y;
		k_index = kx * kernel_size + ky;

		sum += (int)image[index] * kernel[k_index];
	}

	atomicAdd((blurred_image + target_index), sum);
}

/*cv::Mat getGaussianKernel(int rows, int cols, double sigmax, double sigmay)
{
	auto gauss_x = cv::getGaussianKernel(cols, sigmax, CV_32F);
	auto gauss_y = cv::getGaussianKernel(rows, sigmay, CV_32F);
	return gauss_x * gauss_y.t();
}*/


Mat gauss_gpu_v2(Mat input_mat, int kernel_size) {
	Mat gauss_kernel = getGaussianKernel(kernel_size, kernel_size, -1, -1);
	float *result, *result_gpu, *gaussian_kernel_gpu;
	unsigned char* input_mat_gpu;
	int rows = input_mat.rows;
	int cols = input_mat.cols;
	cudaError_t cudaStatus;

	result = new float[rows * cols];
	// Allocate GPU buffers for three arrays (image, kernel, output).
	cudaStatus = cudaMalloc((void**)&input_mat_gpu, rows * cols * sizeof(unsigned char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&result_gpu, rows * cols * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&gaussian_kernel_gpu, kernel_size * kernel_size * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy input matrices from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(input_mat_gpu, input_mat.data, rows * cols * sizeof(unsigned char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(gaussian_kernel_gpu, gauss_kernel.data, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	int threadsInBlock = sqrt(1024/kernel_size);
	dim3 gridDim(int(rows / threadsInBlock) + 1, int(cols / threadsInBlock) + 1, 1);
	dim3 blockDim(threadsInBlock, threadsInBlock, kernel_size);
	printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
		gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	gauss_gpu_kernel_v2 << < gridDim, blockDim >> > (input_mat_gpu, gaussian_kernel_gpu, kernel_size, rows, cols, result_gpu);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "time taken in GPU for " << kernel_size << "sized kernel- " << milliseconds / 1000 << endl;

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(result, result_gpu, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(result_gpu);
	cudaFree(input_mat_gpu);

	Mat result_mat = Mat(rows, cols, CV_32FC1, result);

	return result_mat;
}

Mat gaussian_blur(String filePath, int kernel_size) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// The second parameter determines the color channel of imread. 0 is used to get a grayscale image.
	Mat img = imread(filePath, 0);

	if (img.data == NULL) {
		cout << "Image read failed." << endl;
		return img;
	}

	int rows = img.rows;
	int cols = img.cols;
	cout << "some rows and cols" << img(Range(0, 1), Range(101, 200));

	cudaEventRecord(start);
	Mat blurred_img = gauss_gpu_v2(img, kernel_size);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "time taken for " << kernel_size << "sized kernel- " << milliseconds / 1000 << endl;

	return blurred_img;
}

int main()
{
	int kernel_size = 21;
	String imageName = "Two_lane_city_streets.jpg";

	Mat img = imread(imageName, 0);
	Mat blurred_img = gaussian_blur(imageName, kernel_size);
	Mat normalized_img;

	// the blurred image has returned in float format. It is converted into unsigned character to display properly.
	blurred_img.convertTo(normalized_img, CV_8UC1);
	int rows = blurred_img.rows;
	int cols = blurred_img.cols;
	cout << "Type- " << blurred_img.type() << "result: rows = " << rows << endl << "cols = " << cols << endl;
	cout << "some rows and cols" << normalized_img(Range(0, 1), Range(101, 200));
	cout << endl;
	namedWindow("image", WINDOW_NORMAL);
	ShowManyImages("image", 2, img, normalized_img);
	waitKey(0);

	return 0;
}
