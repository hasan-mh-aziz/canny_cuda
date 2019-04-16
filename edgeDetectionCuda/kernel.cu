
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include<opencv2/opencv.hpp>
#include<iostream>
#include <stdarg.h>
using namespace std;
using namespace cv;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void gauss_gpu_kernel(unsigned char *image, float* kernel, int kernel_size, int rows, int cols, float *blurred_image) {
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	int idy = threadIdx.y + blockIdx.y * blockDim.y;

	if (idx > rows || idy > cols) {
		return;
	}

	int target_index = idx * cols + idy;

	int x, y, index, k_index;

	float sum = 0;
	for (int kx = 0; kx < kernel_size; kx++) {
		x = idx - kernel_size / 2 + kx;
		if (x < 0) continue;
		for (int ky = 0; ky < kernel_size; ky++) {
			y = idy - kernel_size / 2 + ky;
			if (y < 0) continue;

			index = x * cols + y;
			k_index = kx * kernel_size + ky;

			sum += (int)image[index] * kernel[k_index];
		}
	}
	// printf("index- %d, value- %d", target_index, sum);
	blurred_image[target_index] = sum;
}

cv::Mat getGaussianKernel(int rows, int cols, double sigmax, double sigmay)
{
	auto gauss_x = cv::getGaussianKernel(cols, sigmax, CV_32F);
	auto gauss_y = cv::getGaussianKernel(rows, sigmay, CV_32F);
	return gauss_x * gauss_y.t();
}


Mat gauss_gpu(Mat input_mat, int kernel_size) {
	Mat gauss_kernel = getGaussianKernel(kernel_size, kernel_size, -1, -1);
	float *result, *result_gpu, *gaussian_kernel_gpu;
	unsigned char* input_mat_gpu;
	int rows = input_mat.rows;
	int cols = input_mat.cols;
	cudaError_t cudaStatus;

	result = new float[rows * cols];
	for (int i = 0; i < 5; i++) {
		result[i] = 0;
		cout << result[i] << " ";
	}
	cout << endl;
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

	int threadsInBlock = 16;
	dim3 gridDim(int(rows / threadsInBlock) + 1, int(cols / threadsInBlock) + 1, 1);
	dim3 blockDim(threadsInBlock, threadsInBlock, 1);
	printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
		gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
	gauss_gpu_kernel <<< gridDim, blockDim >>> (input_mat_gpu, gaussian_kernel_gpu, kernel_size, rows, cols, result_gpu);
	//printf("time taken for gpu internally--- %s seconds ---" % (time.time() - start_gpu_internal_time));

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

	// short int *result_int = (short int*)result;
	cout << "some rows and cols[";
	for (int i = 101; i < 200; i++) {
		cout << result[i] << ", ";
	}
	cout << ";" << endl;

	
	Mat result_mat = Mat(rows, cols, CV_32FC1, result);

	return result_mat;
}

void ShowManyImages(string title, int nArgs, ...) {
	int size;
	int i;
	int m, n;
	int x, y;

	// w - Maximum number of images in a row
	// h - Maximum number of images in a column
	int w, h;

	// scale - How much we have to resize the image
	float scale;
	int max;

	// If the number of arguments is lesser than 0 or greater than 12
	// return without displaying
	if (nArgs <= 0) {
		printf("Number of arguments too small....\n");
		return;
	}
	else if (nArgs > 14) {
		printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
		return;
	}
	// Determine the size of the image,
	// and the number of rows/cols
	// from number of arguments
	else if (nArgs == 1) {
		w = h = 1;
		size = 300;
	}
	else if (nArgs == 2) {
		w = 2; h = 1;
		size = 300;
	}
	else if (nArgs == 3 || nArgs == 4) {
		w = 2; h = 2;
		size = 300;
	}
	else if (nArgs == 5 || nArgs == 6) {
		w = 3; h = 2;
		size = 200;
	}
	else if (nArgs == 7 || nArgs == 8) {
		w = 4; h = 2;
		size = 200;
	}
	else {
		w = 4; h = 3;
		size = 150;
	}

	// Create a new 3 channel image
	Mat DispImage = Mat::zeros(Size(100 + size * w, 60 + size * h), CV_8UC1);

	// Used to get the arguments passed
	va_list args;
	va_start(args, nArgs);

	// Loop for nArgs number of arguments
	for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
		// Get the Pointer to the IplImage
		Mat img = va_arg(args, Mat);

		// Check whether it is NULL or not
		// If it is NULL, release the image, and return
		if (img.empty()) {
			printf("Invalid arguments");
			return;
		}

		// Find the width and height of the image
		x = img.cols;
		y = img.rows;

		// Find whether height or width is greater in order to resize the image
		max = (x > y) ? x : y;

		// Find the scaling factor to resize the image
		scale = (float)((float)max / size);

		// Used to Align the images
		if (i % w == 0 && m != 20) {
			m = 20;
			n += 20 + size;
		}

		// Set the image ROI to display the current image
		// Resize the input image and copy the it to the Single Big Image
		Rect ROI(m, n, (int)(x / scale), (int)(y / scale));
		Mat temp; resize(img, temp, Size(ROI.width, ROI.height));
		temp.copyTo(DispImage(ROI));
	}

	// Create a new window, and show the Single Big Image
	namedWindow(title, 1);
	imshow(title, DispImage);
	waitKey();

	// End the number of arguments
	va_end(args);
}

int main()
{
	Mat img = imread("Two_lane_city_streets.jpg", 0);
	int rows = img.rows;
	int cols = img.cols;
	cout << "Type- " << img.type() << "original: rows = " << rows << endl << "cols = " << cols << endl;
	cout << "some rows and cols" << img(Range(0, 1), Range(101, 200));

	int kernel_size = 5;
	Mat blurred_img = gauss_gpu(img, kernel_size);
	Mat normalized_img;
	blurred_img.convertTo(normalized_img, CV_8UC1);
	rows = blurred_img.rows;
	cols = blurred_img.cols;
	cout << "Type- " << blurred_img.type() << "result: rows = " << rows << endl << "cols = " << cols << endl;
	cout << "some rows and cols" << normalized_img(Range(0, 1), Range(101, 200));
	cout << endl;
	namedWindow("image", WINDOW_NORMAL);
	ShowManyImages("image", 2, img, normalized_img);
	waitKey(0);
	/*const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    // cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }*/

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

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
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
