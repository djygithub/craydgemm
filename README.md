# craydgemm
c++/cuda/hip double-precision DGEMM CPU/GPU heat/power/smoke test for windows and linux
# linux/rocm/hip
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
# Windows 10 CUDA 10.2.2
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
# Windows 10 oneAPI dpc++
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
