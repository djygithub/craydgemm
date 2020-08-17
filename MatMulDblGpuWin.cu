#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <time.h>
#include <stdlib.h> 
#include <algorithm>
#include <iostream>
#include <windows.h>
using namespace std;

/* #include <sys/time.h> */

/*
CUDA Tutorial, matrix-matrix multiply
UC Berkeley Reactor Design and Neutronics Group
Ryan M. Bergmann - 1/22/2014
C:\dellmatmul\double>nvcc -o MatMulDblGpuWin.exe MatMulDblGpuWin.cu
MatMulDblGpuWin.cu
   Creating library MatMulDblGpuWin.lib and object MatMulDblGpuWin.exp
C:\dellmatmul\double>MatMulDblGpuWin.exe 2000
------ Matrix Dimensions ------
dims a,b = 2000 , 2000
info: allocate host mem ( 91.55 MB)
info: device  mem ( 91.55 MB)
Filling in 2D arrays a and b
Filling Complete
------- CUDA Parameters -------
NUM_THREADS(  16,  16,   0)
       blks( 125, 125,   0)
TOTAL GFLOPS 16.000000
-------------------------------
CPU took 72.563000 seconds as computed by gettickcount
CPU-DOUBLE-GFLOPS/second 0.220498

CPU Matrix multiplication completed. Time to launch GPU kernel.

GPU took 0.632657 seconds as computed by CudaEvent function
GPU-DOUBLE-GFLOPS/second 25.290170

Experiment Done.
-------------------------------

C:\dellmatmul\double>

*/
#define BILLION  1E9

#define CHECK(cmd) \
{\
    cudaError_t error  = cmd;\
    if (error != cudaSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", cudaGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
          }\
}



void matmul_cpu(long len, double* a , double* b , double* c){

	// initialize local variable to hold values while the sum is done
	double sum;
	unsigned row,col,k;

	for(col=0 ; col<len ; col++ ){       //scan the rows
		for(row=0 ; row<len ; row++ ){   //scan the cols

			// zero out sum
			sum = 0;

			// scan the row of a, the col of b
			for(k=0;k<len;k++){
				sum +=  a[ row * len + k ] * b[ k * len + col ];
			}

			// write final value into output array
			c[ len * row + col  ] = sum;

		}
	}


}

__global__ void matmul_kernel( long len, double* a , double* b , double* c){

	//
	//  THIS IS THE SIMPLE WAY TO DO IT, NOT THE ***FAST WAY*** -> uses 2*N^3 global loads
	//

	// get index in c
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	//return if over the length
	if(row>=len | col>=len){return;}

	// initialize local variable to hold values while the sum is done
	double sum = 0;
	unsigned j;

	// scan the row of a, the col of b
	for(j=0;j<len;j++){
		sum +=  a[ row * len + j ] * b[ j * len + col ];
	}

	// write final value into output array
	c[ len * row + col  ] = sum;

}


int get_time(){

	return ((int)clock())/((int)CLOCKS_PER_SEC);

}

int main(int argc, char *argv[]){

	// declare
	double* 		a;
	double*  	b;  
	double* 		c;
	double* 		d_a;
	double*		d_b;
	double*		d_c;
	long            N = atol(argv[1]);
        double          matrixsize = N;
	long  	        len_a=N, len_b=N, j, k;
	long 		bytes_a, bytes_b, bytes_c;
	dim3 		NUM_THREADS, blks;
        double gflops = ((matrixsize /1000) * (matrixsize / 1000) * (matrixsize / 1000) * 2);
        double cpugiops;
        double gpugiops;
	cudaEvent_t 	start, startcpu, stop, stopcpu;
	float		time;
  
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
        cudaEventCreate(&startcpu);
	cudaEventCreate(&stopcpu);


	printf("------ Matrix Dimensions ------\n");
	printf("dims a,b = %ld , %ld \n",len_a,len_b);
	assert(len_a==len_b);
	bytes_a = len_a * len_a * sizeof(double);
	bytes_b = len_b * len_b * sizeof(double);
	bytes_c = len_b * len_b * sizeof(double);
        
        printf ("info: allocate host mem (%6.2f MB)\n", (3.0*bytes_a)/1024.0/1024.0);
	//allocate arrays
	//a = (int*) malloc( bytes_a );
	cudaHostAlloc(&a,bytes_a,cudaHostAllocDefault);
	CHECK(a == 0 ? cudaErrorMemoryAllocation : cudaSuccess );
	//b = (int*) malloc( bytes_b );
	cudaHostAlloc(&b,bytes_b,cudaHostAllocDefault);
	CHECK(b == 0 ? cudaErrorMemoryAllocation : cudaSuccess );
	//c = (int*) malloc( bytes_b );
	cudaHostAlloc(&c,bytes_c,cudaHostAllocDefault);
	CHECK(c == 0 ? cudaErrorMemoryAllocation : cudaSuccess );
        printf ("info: device  mem (%6.2f MB)\n", (3.0*bytes_a)/1024.0/1024.0);
	//allocate device arrays
	CHECK(cudaMalloc( &d_a , bytes_a ));  //must be pointer to the point, since the actual point value is being changed, not the value it points to
	CHECK(cudaMalloc( &d_b , bytes_b ));
	CHECK(cudaMalloc( &d_c , bytes_c ));
         
	if(a==NULL || b == NULL || c == NULL )
	printf("Could not allocate host memory\n");
        printf("Filling in 2D arrays a and b \n");
	// read in data
	for(j=0;j<len_a;j++){
		for(k=0;k<len_a;k++){
			a[j*len_a+k]= 2;  //row major
			b[j*len_a+k] = 2;
		}
	}

        printf("Filling Complete\n");
	// determine gpu parameters, print them
	NUM_THREADS.x   = NUM_THREADS.y = 16;
	blks.x = blks.y = (len_a + NUM_THREADS.x - 1 ) / NUM_THREADS.x;
	NUM_THREADS.z   = blks.z = 1;
	printf("------- CUDA Parameters -------\n");
	printf("NUM_THREADS(%4u,%4u,   0)\n       blks(%4u,%4u,   0)\n",NUM_THREADS.x,NUM_THREADS.y,blks.x,blks.y);
        printf("TOTAL GFLOPS %lf \n",gflops);
	printf("-------------------------------\n");

	//launch cpu version to compare
  
	 /* gettimeofday (&tvalBefore, NULL); */
   	
        DWORD dw1 = GetTickCount();
	matmul_cpu(len_a, a, b, c);
        DWORD dw2 = GetTickCount();
        double result = dw2-dw1;
        printf("CPU took %f seconds as computed by gettickcount\n", result/1000);
        cpugiops = gflops / (result/1000);
        printf ("CPU-DOUBLE-GFLOPS/second %lf \n",cpugiops); 
        printf("\nCPU Matrix multiplication completed. Time to launch GPU kernel.\n");
	cudaEventRecord(start, 0);

        cudaMemcpy( d_a , a , bytes_a , cudaMemcpyHostToDevice );
        cudaMemcpy( d_b , b , bytes_b , cudaMemcpyHostToDevice );


        matmul_kernel<<< blks, NUM_THREADS>>> (len_a , d_a , d_b , d_c);
        cudaMemcpy( b , d_c , bytes_b , cudaMemcpyDeviceToHost );
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("\nGPU took %f seconds as computed by CudaEvent function\n", time/1000);
        gpugiops = gflops / (time/1000);
        printf ("GPU-DOUBLE-GFLOPS/second %lf \n",gpugiops);

	if(cudaPeekAtLastError()){
		printf("CUDA ERROR, %s\n",cudaGetErrorString(cudaPeekAtLastError()));
		return 1;
	}


	printf("\nExperiment Done.\n");
	printf("-------------------------------\n");
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	// return zero if all ok
	return 0;

}
