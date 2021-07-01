# oneAPI/c++/cuda/hip DGEMM CPU/GPU heat/power/smoke test for windows and linux
Based on a tool from Dell which is based on a tutorial by Ryan Bergmann UC Berkeley
## linux/rocm/hip
Copy craydgemm.zip from github (djygithub/craydgemm) to test directory, unzip it.  
```
abc@djy:~/craydgemm$ unzip craydgemm.zip
Archive:  craydgemm.zip
  inflating: craydgemm/MatMulCompilePrep.sh
  inflating: craydgemm/Matrix_Multiplication_GPU_DOUBLE_GIOPS_PINNED_TWO.cu
 extracting: craydgemm/mmdblgpu00.loop.sh
 extracting: craydgemm/mmdblgpu01.loop.sh
 extracting: craydgemm/mmdblgpu02.loop.sh
 extracting: craydgemm/mmdblgpu03.loop.sh
 extracting: craydgemm/mmdblgpu04.loop.sh
 extracting: craydgemm/mmdblgpu05.loop.sh
 extracting: craydgemm/mmdblgpu06.loop.sh
 extracting: craydgemm/mmdblgpu07.loop.sh
  inflating: craydgemm/rocmsmi.bw.sh
  inflating: craydgemm/rocmsmi.sh
  inflating: craydgemm/test4.sh
  inflating: craydgemm/test8.sh
abc@djy:~/craydgemm$  
```
Use chmod to mark MatMulCompilePrep.sh executable, it will hipify, compile, then execute mmdblgpugiops 10 times varying the matrix size from 1,000 doubles to 10,000 doubles as a warm up.  This script also marks the *.sh files executable.
```
abc@djy:~/craydgemm$ cd craydgemm
abc@djy:~/craydgemm/craydgemm$ chmod +x ./MatMulCompilePrep.sh
abc@djy:~/craydgemm/craydgemm$ ./MatMulCompilePrep.sh
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MULTIPLYING TWO SQUARE MATRICES OF SIZE  1000 FLOATS
------ Matrix Dimensions ------
dims a,b = 1000 , 1000
info: allocate host mem ( 22.89 MB)
info: device  mem ( 22.89 MB)
Filling in 2D arrays a and b
Filling Complete
------- CUDA Parameters -------
NUM_THREADS(  16,  16,   0)
       blks(  63,  63,   0)
TOTAL DBLOPS 2000000000.000000
-------------------------------

Calling CPU Matrix Multiply

CPU took 0.000000 seconds as computed by gettimeofday() function

CPU Matrix multiplication completed. Time to launch GPU kernel.

GPU took 0.371508 seconds as computed by CudaEvent function
GPU-GDBLOPS/second 5.383459

Experiment Done.
-------------------------------
============================================================================================================
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
MULTIPLYING TWO SQUARE MATRICES OF SIZE  2000 FLOATS
```
There are eight scripts provided for exercising eight gpus (0, 1, 2, 3, 4, 5, 6, 7), each looks like the following with a matrix size of 25000 doubles.
```
abc@djy:~/craydgemm/craydgemm$ cat mmdblgpu00.loop.sh
export HIP_VISIBLE_DEVICES=0
while true
do
date
./mmdblgpugiops 25000
done
abc@djy:~/craydgemm/craydgemm$
```


Start an eight GPU stress test by executing ./test8 
```
abc@djy:~/craydgemm/craydgemm$ ./test8.sh
Wed Sep 25 07:37:43 PDT 2019
Wed Sep 25 07:37:43 PDT 2019
Wed Sep 25 07:37:43 PDT 2019
abc@djy:~/craydgemm/craydgemm$
Wed Sep 25 07:37:43 PDT 2019
Wed Sep 25 07:37:43 PDT 2019
Wed Sep 25 07:37:43 PDT 2019
Wed Sep 25 07:37:43 PDT 2019
Wed Sep 25 07:37:43 PDT 2019
------ Matrix Dimensions ------
dims a,b = 25000 , 25000
info: allocate host mem (14305.11 MB)
------ Matrix Dimensions ------
dims a,b = 25000 , 25000
info: allocate host mem (14305.11 MB)
------ Matrix Dimensions ------
dims a,b = 25000 , 25000
info: allocate host mem (14305.11 MB)
------ Matrix Dimensions ------
dims a,b = 25000 , 25000
info: allocate host mem (14305.11 MB)
------ Matrix Dimensions ------
dims a,b = 25000 , 25000
info: allocate host mem (14305.11 MB)
------ Matrix Dimensions ------
dims a,b = 25000 , 25000
info: allocate host mem (14305.11 MB)
------ Matrix Dimensions ------
dims a,b = 25000 , 25000
info: allocate host mem (14305.11 MB)
------ Matrix Dimensions ------
dims a,b = 25000 , 25000
info: allocate host mem (14305.11 MB)
```
Start a rocmsmi.sh loop in another window to view GPU activity
```
abc@djy:~/craydgemm/craydgemm$ ./rocmsmi.sh
Wed Sep 25 07:37:25 PDT 2019


========================ROCm System Management Interface========================
================================================================================
GPU  Temp   AvgPwr  SCLK     MCLK     Fan     Perf  PwrCap  VRAM%  GPU%
1    41.0c  N/A     1606Mhz  1000Mhz  0.0%    auto  225.0W   88%   100%
2    41.0c  N/A     1606Mhz  1000Mhz  16.86%  auto  225.0W   88%   100%
3    38.0c  N/A     1725Mhz  1000Mhz  5.88%   auto  225.0W   88%   100%
4    38.0c  N/A     1725Mhz  1000Mhz  11.76%  auto  225.0W   88%   100%
5    41.0c  N/A     1725Mhz  1000Mhz  4.71%   auto  225.0W   88%   100%
6    42.0c  N/A     1725Mhz  1000Mhz  0.0%    auto  225.0W   88%   100%
7    37.0c  N/A     1606Mhz  1000Mhz  17.65%  auto  225.0W   88%   100%
8    42.0c  N/A     1606Mhz  1000Mhz  18.82%  auto  225.0W   88%   100%
================================================================================
==============================End of ROCm SMI Log ==============================
```
Use kill -9 -1 to kill the MatMul processes and all the tasks you can for your userid.  This will also kill any terminal sessions for your userid.
## Windows 10 Cuda 10.2.2 Intel i7-7700 Nvidia gtx1050
Windows/NVCC If you run into an issue finding the cl.exe executable here's a workaround
```
C:\dellmatmul>nvcc -o MatMulDblGpuWin.exe MatMulDblGpuWin.cu
nvcc fatal   : Cannot find compiler 'cl.exe' in PATH

C:\dellmatmul>nvcc -o MatMulDblGpuWin.exe MatMulDblGpuWin.cu -ccbin "c:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin"
MatMulDblGpuWin.cu
   Creating library MatMulDblGpuWin.lib and object MatMulDblGpuWin.exp

C:\dellmatmul>dir
 Volume in drive C is OS
 Volume Serial Number is ECEE-F446

 Directory of C:\dellmatmul

11/15/2020  07:36 PM    <DIR>          .
11/15/2020  07:36 PM    <DIR>          ..
11/07/2020  07:13 AM    <DIR>          fromgithub
11/07/2020  07:09 AM             6,599 MatMulDblGpuWin.cu
11/15/2020  07:36 PM           295,424 MatMulDblGpuWin.exe
11/15/2020  07:36 PM               663 MatMulDblGpuWin.exp
11/15/2020  07:36 PM             1,898 MatMulDblGpuWin.lib
08/10/2020  11:26 AM           295,424 mm.exe
08/10/2020  11:26 AM               638 mm.exp
08/10/2020  11:26 AM             1,664 mm.lib
               7 File(s)        602,310 bytes
               3 Dir(s)  581,298,413,568 bytes free

C:\dellmatmul>
```
Execute
```
c:\dellmatmul\cuda10-2-2>MatMulDblGpuWin.exe 1000
------ Matrix Dimensions ------
dims a,b = 1000 , 1000
info: allocate host mem ( 22.89 MB)
info: device  mem ( 22.89 MB)
Filling in 2D arrays a and b
Filling Complete
------- CUDA Parameters -------
NUM_THREADS(  16,  16,   0)
       blks(  63,  63,   0)
TOTAL GFLOPS 2.000000
-------------------------------
CPU took 3.485000 seconds as computed by gettickcount
CPU-DOUBLE-GFLOPS/second 0.573888

CPU Matrix multiplication completed. Time to launch GPU kernel.

GPU took 0.041469 seconds as computed by CudaEvent function
GPU-DOUBLE-GFLOPS/second 48.228879

Experiment Done.
-------------------------------

c:\dellmatmul\cuda10-2-2>MatMulDblGpuWin.exe 2000
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
CPU took 42.172000 seconds as computed by gettickcount
CPU-DOUBLE-GFLOPS/second 0.379399

CPU Matrix multiplication completed. Time to launch GPU kernel.

GPU took 0.278111 seconds as computed by CudaEvent function
GPU-DOUBLE-GFLOPS/second 57.530996

Experiment Done.
-------------------------------

c:\dellmatmul\cuda10-2-2>MatMulDblGpuWin.exe 3000
------ Matrix Dimensions ------
dims a,b = 3000 , 3000
info: allocate host mem (205.99 MB)
info: device  mem (205.99 MB)
Filling in 2D arrays a and b
Filling Complete
------- CUDA Parameters -------
NUM_THREADS(  16,  16,   0)
       blks( 188, 188,   0)
TOTAL GFLOPS 54.000000
-------------------------------
CPU took 154.031000 seconds as computed by gettickcount
CPU-DOUBLE-GFLOPS/second 0.350579

CPU Matrix multiplication completed. Time to launch GPU kernel.

GPU took 0.945665 seconds as computed by CudaEvent function
GPU-DOUBLE-GFLOPS/second 57.102688

Experiment Done.
-------------------------------

c:\dellmatmul\cuda10-2-2>
```
## Windows 10 cuda 10.2.2 oneAPI dpc++ Intel I7-7700 Intel HD-630
Initialize oneAPI environment
```
c:\Program Files (x86)\Intel\oneAPI>setvars.bat
:: initializing oneAPI environment...
   initializing Visual Studio command-line environment...
   Visual Studio version 16.10.0 environment configured.
   Visual Studio environment initialized for: 'x64'
:  advisor -- latest
:  compiler -- latest
:  dal -- latest
:  debugger -- latest
:  dev-utilities -- latest
:  dnnl -- latest
:  dpcpp-ct -- latest
:  dpl -- latest
:  inspector -- latest
:  intelpython -- latest
failed to create process.
:  ipp -- latest
:  ippcp -- latest
:  itac -- latest
:  mkl -- latest
:  mpi -- latest
:  tbb -- latest
:  vpl -- latest
:  vtune -- latest
:: oneAPI environment initialized ::

c:\Program Files (x86)\Intel\oneAPI>
```
 Intel® DPC++ Compatibility Tool
```
c:\dellmatmul\cuda10-2-2>dpct MatMulDblGpuWin.cu
NOTE: Could not auto-detect compilation database for file 'MatMulDblGpuWin.cu' in 'c:\dellmatmul\cuda10-2-2' or any parent directory.
The directory "dpct_output" is used as "out-root"
Processing: c:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:150:27: warning: DPCT1048:0: The original value cudaHostAllocDefault is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
        cudaHostAlloc(&a,bytes_a,cudaHostAllocDefault);
                                 ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:153:27: warning: DPCT1048:1: The original value cudaHostAllocDefault is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
        cudaHostAlloc(&b,bytes_b,cudaHostAllocDefault);
                                 ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:156:27: warning: DPCT1048:2: The original value cudaHostAllocDefault is not meaningful in the migrated code and was removed or replaced with 0. You may need to check the migrated code.
        cudaHostAlloc(&c,bytes_c,cudaHostAllocDefault);
                                 ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:110:15: warning: DPCT1008:3: clock function is not defined in the DPC++. This is a hardware-specific feature. Consult with your hardware vendor to find a replacement.
        return ((int)clock())/((int)CLOCKS_PER_SEC);
                     ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:134:2: warning: DPCT1026:4: The call to cudaEventCreate was removed, because this call is redundant in DPC++.
        cudaEventCreate(&start);
        ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:135:2: warning: DPCT1026:5: The call to cudaEventCreate was removed, because this call is redundant in DPC++.
        cudaEventCreate(&stop);
        ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:136:9: warning: DPCT1026:6: The call to cudaEventCreate was removed, because this call is redundant in DPC++.
        cudaEventCreate(&startcpu);
        ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:137:2: warning: DPCT1026:7: The call to cudaEventCreate was removed, because this call is redundant in DPC++.
        cudaEventCreate(&stopcpu);
        ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:197:2: warning: DPCT1012:8: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
        cudaEventRecord(start, 0);
        ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:203:9: warning: DPCT1049:9: The workgroup size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the workgroup size if needed.
        matmul_kernel<<< blks, NUM_THREADS>>> (len_a , d_a , d_b , d_c);
        ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:205:2: warning: DPCT1012:10: Detected kernel execution time measurement pattern and generated an initial code for time measurements in SYCL. You can change the way time is measured depending on your goals.
        cudaEventRecord(stop, 0);
        ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:47:55: warning: DPCT1009:11: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", cudaGetErrorString(error), error,__FILE__, __LINE__); \
                                                      ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:160:8: warning: DPCT1003:12: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
        CHECK(cudaMalloc( &d_a , bytes_a ));  //must be pointer to the point, since the actual point value is being changed, not the value it points to
              ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:161:8: warning: DPCT1003:13: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
        CHECK(cudaMalloc( &d_b , bytes_b ));
              ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:162:8: warning: DPCT1003:14: Migrated API does not return error code. (*, 0) is inserted. You may need to rewrite this code.
        CHECK(cudaMalloc( &d_c , bytes_c ));
              ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:213:5: warning: DPCT1010:15: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
        if(cudaPeekAtLastError()){
           ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:214:29: warning: DPCT1009:16: SYCL uses exceptions to report errors and does not use the error codes. The original code was commented out and a warning string was inserted. You need to rewrite this code.
                printf("CUDA ERROR, %s\n",cudaGetErrorString(cudaPeekAtLastError()));
                                          ^
C:\dellmatmul\cuda10-2-2\MatMulDblGpuWin.cu:214:48: warning: DPCT1010:17: SYCL uses exceptions to report errors and does not use the error codes. The call was replaced with 0. You need to rewrite this code.
                printf("CUDA ERROR, %s\n",cudaGetErrorString(cudaPeekAtLastError()));
                                                             ^
Processed 1 file(s) in -in-root folder "C:\dellmatmul\cuda10-2-2"

See Diagnostics Reference to resolve warnings and complete the migration:
https://software.intel.com/content/www/us/en/develop/documentation/intel-dpcpp-compatibility-tool-user-guide/top/diagnostics-reference.html
```
Compile with dpc++
```
c:\dellmatmul\cuda10-2-2\dpct_output>dpcpp matmuldblgpuwin.dp.cpp /EHsc
matmuldblgpuwin.dp.cpp(233,16): warning: format specifies type 'unsigned int' but the argument has type 'size_t'
      (aka 'unsigned long long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
               ^~~~~~~~~~~~~~
matmuldblgpuwin.dp.cpp(233,32): warning: format specifies type 'unsigned int' but the argument has type 'size_t'
      (aka 'unsigned long long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
                               ^~~~~~~~~~~~~~
matmuldblgpuwin.dp.cpp(233,48): warning: format specifies type 'unsigned int' but the argument has type 'size_t'
      (aka 'unsigned long long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
                                               ^~~~~~~
matmuldblgpuwin.dp.cpp(233,57): warning: format specifies type 'unsigned int' but the argument has type 'size_t'
      (aka 'unsigned long long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
                                                        ^~~~~~~
4 warnings generated.
matmuldblgpuwin.dp.cpp(233,16): warning: format specifies type 'unsigned int' but the argument has type 'size_t'
      (aka 'unsigned long long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
               ^~~~~~~~~~~~~~
matmuldblgpuwin.dp.cpp(233,32): warning: format specifies type 'unsigned int' but the argument has type 'size_t'
      (aka 'unsigned long long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
                               ^~~~~~~~~~~~~~
matmuldblgpuwin.dp.cpp(233,48): warning: format specifies type 'unsigned int' but the argument has type 'size_t'
      (aka 'unsigned long long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
                                               ^~~~~~~
matmuldblgpuwin.dp.cpp(233,57): warning: format specifies type 'unsigned int' but the argument has type 'size_t'
      (aka 'unsigned long long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
                                                        ^~~~~~~
4 warnings generated.
matmuldblgpuwin.dp.cpp(233,16): warning: format specifies type 'unsigned int' but the argument has type 'size_t'
      (aka 'unsigned long long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
               ^~~~~~~~~~~~~~
matmuldblgpuwin.dp.cpp(233,32): warning: format specifies type 'unsigned int' but the argument has type 'size_t'
      (aka 'unsigned long long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
                               ^~~~~~~~~~~~~~
matmuldblgpuwin.dp.cpp(233,48): warning: format specifies type 'unsigned int' but the argument has type 'size_t'
      (aka 'unsigned long long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
                                               ^~~~~~~
matmuldblgpuwin.dp.cpp(233,57): warning: format specifies type 'unsigned int' but the argument has type 'size_t'
      (aka 'unsigned long long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
                                                        ^~~~~~~
4 warnings generated.

c:\dellmatmul\cuda10-2-2\dpct_output>dir
 Volume in drive C is OS
 Volume Serial Number is ECEE-F446

 Directory of c:\dellmatmul\cuda10-2-2\dpct_output

06/30/2021  05:13 PM    <DIR>          .
06/30/2021  05:13 PM    <DIR>          ..
06/30/2021  05:08 PM    <DIR>          dpct_output
06/30/2021  05:02 PM            35,396 MainSourceFiles.yaml
06/30/2021  05:02 PM            11,127 matmuldblgpuwin.dp.cpp
06/30/2021  05:13 PM            87,552 matmuldblgpuwin.dp.exe
               3 File(s)        134,075 bytes
               3 Dir(s)  316,539,334,656 bytes free


```
Execute
```
c:\dellmatmul\cuda10-2-2\dpct_output>matmuldblgpuwin.dp.exe 1000
------ Matrix Dimensions ------
dims a,b = 1000 , 1000
info: allocate host mem ( 22.89 MB)
info: device  mem ( 22.89 MB)
Filling in 2D arrays a and b
Filling Complete
------- CUDA Parameters -------
NUM_THREADS(  16,  16,   0)
       blks(  63,  63,   0)
TOTAL GFLOPS 2.000000
-------------------------------
CPU took 1.359000 seconds as computed by gettickcount
CPU-DOUBLE-GFLOPS/second 1.471670

CPU Matrix multiplication completed. Time to launch GPU kernel.

GPU took 0.482100 seconds as computed by CudaEvent function
GPU-DOUBLE-GFLOPS/second 4.148514

Experiment Done.
-------------------------------

c:\dellmatmul\cuda10-2-2\dpct_output>matmuldblgpuwin.dp.exe 2000
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
CPU took 32.469000 seconds as computed by gettickcount
CPU-DOUBLE-GFLOPS/second 0.492778

CPU Matrix multiplication completed. Time to launch GPU kernel.

GPU took 1.254604 seconds as computed by CudaEvent function
GPU-DOUBLE-GFLOPS/second 12.753027

Experiment Done.
-------------------------------

c:\dellmatmul\cuda10-2-2\dpct_output>matmuldblgpuwin.dp.exe 3000
------ Matrix Dimensions ------
dims a,b = 3000 , 3000
info: allocate host mem (205.99 MB)
info: device  mem (205.99 MB)
Filling in 2D arrays a and b
Filling Complete
------- CUDA Parameters -------
NUM_THREADS(  16,  16,   0)
       blks( 188, 188,   0)
TOTAL GFLOPS 54.000000
-------------------------------
CPU took 135.547000 seconds as computed by gettickcount
CPU-DOUBLE-GFLOPS/second 0.398386

CPU Matrix multiplication completed. Time to launch GPU kernel.

GPU took 4.119520 seconds as computed by CudaEvent function
GPU-DOUBLE-GFLOPS/second 13.108323

Experiment Done.
-------------------------------
```
