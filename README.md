# oneAPI/c++/cuda/hip DGEMM CPU/GPU heat/power/smoke test for windows and linux provides double precision GFLOPS/second for CPU and GPU http://davidjyoung.com/cmg/oneAPI.pdf
Based on a tool from Dell, which is based on an integer tutorial by Ryan Bergmann UC Berkeley.
![results](http://davidjyoung.com/cmg/craydgemm2.jpg)<br/>
[rocm linux ryzen r9nano](https://github.com/djygithub/craydgemm#rocm-linux-ryzen-r9nano)<br/>
[cuda windows I7 gtx1050](https://github.com/djygithub/craydgemm#cuda-windows-i7-gtx1050)<br/>
[cuda linux I7 gtx1050](https://github.com/djygithub/craydgemm#cuda-linux-i7-gtx1050)<br/>
[oneApi windows I7 hd630](https://github.com/djygithub/craydgemm#oneapi-windows-i7-hd-630)<br/>
[oneApi linux I7 gtx1050](https://github.com/djygithub/craydgemm#oneapi-linux-i7-gtx1050)<br/>
## rocm linux ryzen R9nano
http://davidjyoung.com/movies/ryzenradeontopcraydgemm/<br/>
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

Use rocprof to gather basic data movement,kernel, and hip statistics. 
https://www.olcf.ornl.gov/wp-content/uploads/2021/04/SPOCK_Libraries_profiling_JMaia.pdf
```
david@ryzen:~/craydgemm/craydgemm$ rocprof --stats ./mmdblgpugiops 3000
RPL: on '220101_123501' from '/opt/rocm-4.1.0/rocprofiler' in '/home/david/craydgemm/craydgemm'
RPL: profiling '"./mmdblgpugiops" "3000"'
RPL: input file ''
RPL: output dir '/tmp/rpl_data_220101_123501_1914'
RPL: result dir '/tmp/rpl_data_220101_123501_1914/input_results_220101_123501'
ROCProfiler: input from "/tmp/rpl_data_220101_123501_1914/input.xml"
  0 metrics
  0 traces
------ Matrix Dimensions ------
dims a,b = 3000 , 3000
info: allocate host mem (205.99 MB)
info: device  mem (205.99 MB)
Filling in 2D arrays a and b
Filling Complete
------- CUDA Parameters -------
NUM_THREADS(  16,  16,   0)
       blks( 188, 188,   0)
TOTAL DBLOPS 54000000000.000000
-------------------------------

Calling CPU Matrix Multiply

CPU took 0.000001 seconds as computed by gettimeofday() function

CPU Matrix multiplication completed. Time to launch GPU kernel.

GPU took 0.464849 seconds as computed by CudaEvent function
GPU-GDBLOPS/second 116.166786

Experiment Done.
-------------------------------

ROCPRofiler: 1 contexts collected, output directory /tmp/rpl_data_220101_123501_1914/input_results_220101_123501
File '/home/david/craydgemm/craydgemm/results.csv' is generating
File '/home/david/craydgemm/craydgemm/results.stats.csv' is generating
```
Use rocprof to gather hip-trace statistics
```
david@ryzen:~/craydgemm/craydgemm$ rocprof --hip-trace ./mmdblgpugiops 3000
RPL: on '220101_123524' from '/opt/rocm-4.1.0/rocprofiler' in '/home/david/craydgemm/craydgemm'
RPL: profiling '"./mmdblgpugiops" "3000"'
RPL: input file ''
RPL: output dir '/tmp/rpl_data_220101_123524_1975'
RPL: result dir '/tmp/rpl_data_220101_123524_1975/input_results_220101_123524'
ROCTracer (pid=1997):
    HIP-trace()
------ Matrix Dimensions ------
dims a,b = 3000 , 3000
info: allocate host mem (205.99 MB)
info: device  mem (205.99 MB)
Filling in 2D arrays a and b
Filling Complete
------- CUDA Parameters -------
NUM_THREADS(  16,  16,   0)
       blks( 188, 188,   0)
TOTAL DBLOPS 54000000000.000000
-------------------------------

Calling CPU Matrix Multiply

CPU took 0.000000 seconds as computed by gettimeofday() function

CPU Matrix multiplication completed. Time to launch GPU kernel.

GPU took 0.459683 seconds as computed by CudaEvent function
GPU-GDBLOPS/second 117.472379

Experiment Done.
-------------------------------
START timestamp found (141387932775ns)
scan ops data 5:6                                                                                                    File '/home/david/                             craydgemm/craydgemm/results.copy_stats.csv' is generating
dump json 2:3
File '/home/david/craydgemm/craydgemm/results.json' is generating
File '/home/david/craydgemm/craydgemm/results.hip_stats.csv' is generating
dump json 21:22
File '/home/david/craydgemm/craydgemm/results.json' is generating
File '/home/david/craydgemm/craydgemm/results.stats.csv' is generating
dump json 2:3
File '/home/david/craydgemm/craydgemm/results.json' is generating
```
Display rocprof statistics at command line
```
david@ryzen:~/craydgemm/craydgemm$ cat results.copy_stats.csv
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
CopyHostToDevice,2,24870831,12435415,68.17713214183681
CopyDeviceToHost,1,11608895,11608895,31.82286785816319
david@ryzen:~/craydgemm/craydgemm$ cat results.stats.csv
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
matmul_kernel(long, double*, double*, double*),1,421948927,421948927,100.0
"<barrier packet>",2,0,0,0.0
david@ryzen:~/craydgemm/craydgemm$ cat results.hip_stats.csv
"Name","Calls","TotalDurationNs","AverageNs","Percentage"
hipMemcpy,3,463859920,154619973,60.46196512966549
hipEventRecord,2,261542935,130771467,34.09089497510448
hipHostMalloc,3,39756067,13252022,5.182016882697356
hipMalloc,3,1471299,490433,0.19177692445019112
hipLaunchKernel,1,481273,481273,0.06273167844259857
hipFree,3,68759,22919,0.008962413179286258
__hipPushCallConfiguration,1,4098,4098,0.0005341550809161722
hipEventCreate,2,3637,1818,0.0004740658929458561
hipEventSynchronize,1,2875,2875,0.0003747427666261579
hipEventElapsedTime,1,992,992,0.00012930254764979084
__hipPopCallConfiguration,1,681,681,8.876515619910036e-05
hipPeekAtLastError,1,391,391,5.096501626115748e-05
david@ryzen:~/craydgemm/craydgemm$

```
Use chrome tracing to display results.json
![chrome tracing](http://davidjyoung.com/cmg/rocprof.json.JPG)
## cuda windows I7 gtx1050
Windows/NVCC If you run into an issue finding the cl.exe executable here's a workaround
```
c:\dellmatmul\cuda10-2-2>nvcc -o MatMulDblGpuWin.exe MatMulDblGpuWin.cu
nvcc fatal   : Cannot find compiler 'cl.exe' in PATH

c:\dellmatmul\cuda10-2-2>nvcc -o MatMulDblGpuWin.exe MatMulDblGpuWin.cu -ccbin "c:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.16.27023\bin\HostX64"
matmuldblgpuwin.cu
   Creating library MatMulDblGpuWin.lib and object MatMulDblGpuWin.exp

c:\dellmatmul\cuda10-2-2>dir
 Volume in drive C is OS
 Volume Serial Number is ECEE-F446

 Directory of c:\dellmatmul\cuda10-2-2

02/01/2022  09:31 PM    <DIR>          .
02/01/2022  09:31 PM    <DIR>          ..
09/10/2021  11:24 AM    <DIR>          20210910inteladvisor
01/01/2022  04:19 PM    <DIR>          dpct_output
11/07/2020  07:09 AM             6,599 MatMulDblGpuWin.cu
02/01/2022  09:31 PM           335,872 MatMulDblGpuWin.exe
02/01/2022  09:31 PM               732 MatMulDblGpuWin.exp
02/01/2022  09:12 PM             1,898 MatMulDblGpuWin.lib
02/01/2022  08:54 PM             6,603 MatMulDblGpuWinNocpu.cu
02/01/2022  09:30 PM           335,872 MatMulDblGpuWinNocpu.exe
02/01/2022  09:30 PM               743 MatMulDblGpuWinNocpu.exp
02/01/2022  09:14 PM             1,966 MatMulDblGpuWinNocpu.lib
07/05/2021  12:19 PM    <DIR>          r000hpc
07/05/2021  12:20 PM    <DIR>          r001hpc
07/05/2021  12:20 PM    <DIR>          r002hpc
07/05/2021  12:22 PM    <DIR>          r003hpc
07/05/2021  12:26 PM    <DIR>          r004hpc
08/15/2021  08:23 AM            45,568 TimeMem.exe
               9 File(s)        735,853 bytes
               9 Dir(s)  105,720,381,440 bytes free

c:\dellmatmul\cuda10-2-2>
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
Use NVPROF to gather kernel times and API statistics 
```
c:\dellmatmul\cuda10-2-2>nvprof MatMulDblGpuWin.exe 3000
==15432== NVPROF is profiling process 15432, command: MatMulDblGpuWin.exe 3000
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
CPU took 155.235000 seconds as computed by gettickcount
CPU-DOUBLE-GFLOPS/second 0.347860

CPU Matrix multiplication completed. Time to launch GPU kernel.

GPU took 0.969663 seconds as computed by CudaEvent function
GPU-DOUBLE-GFLOPS/second 55.689426

Experiment Done.
-------------------------------
==15432== Profiling application: MatMulDblGpuWin.exe 3000
==15432== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   98.28%  952.81ms         1  952.81ms  952.81ms  952.81ms  matmul_kernel(long, double*, double*, double*)
                    1.14%  11.006ms         2  5.5032ms  5.4693ms  5.5370ms  [CUDA memcpy HtoD]
                    0.58%  5.6509ms         1  5.6509ms  5.6509ms  5.6509ms  [CUDA memcpy DtoH]
      API calls:   79.27%  969.71ms         3  323.24ms  5.5178ms  958.52ms  cudaMemcpy
                   10.90%  133.30ms         4  33.325ms     500ns  133.30ms  cudaEventCreate
                    4.77%  58.369ms         1  58.369ms  58.369ms  58.369ms  cuDevicePrimaryCtxRelease
                    4.11%  50.332ms         3  16.777ms  16.504ms  17.008ms  cudaHostAlloc
                    0.87%  10.600ms         3  3.5333ms  3.4948ms  3.5529ms  cudaMalloc
                    0.05%  648.90us         3  216.30us  158.50us  305.80us  cudaFree
                    0.01%  113.00us         1  113.00us  113.00us  113.00us  cuModuleUnload
                    0.01%  75.600us         1  75.600us  75.600us  75.600us  cudaLaunchKernel
                    0.00%  47.300us         2  23.650us  9.6000us  37.700us  cudaEventRecord
                    0.00%  44.500us         1  44.500us  44.500us  44.500us  cudaEventSynchronize
                    0.00%  17.500us         1  17.500us  17.500us  17.500us  cudaEventElapsedTime
                    0.00%  15.300us         1  15.300us  15.300us  15.300us  cuDeviceTotalMem
                    0.00%  14.800us        97     152ns     100ns  1.0000us  cuDeviceGetAttribute
                    0.00%  7.2000us         3  2.4000us     300ns  5.9000us  cuDeviceGetCount
                    0.00%  2.3000us         1  2.3000us  2.3000us  2.3000us  cuDeviceGetName
                    0.00%  1.9000us         2     950ns     200ns  1.7000us  cuDeviceGet
                    0.00%  1.1000us         1  1.1000us  1.1000us  1.1000us  cudaPeekAtLastError
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid

c:\dellmatmul\cuda10-2-2>
```
Use NVPROF to capture double precision flops, memory reads, and memory writes
```
c:\dellmatmul\cuda10-2-2>nvprof --profile-child-processes --metrics flop_count_dp  --metrics dram_read_transactions  --metrics dram_write_transactions MatMulDblGpuWin.exe 3000
==14804== NVPROF is profiling process 14804, command: MatMulDblGpuWin.exe 3000
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
CPU took 154.156000 seconds as computed by gettickcount
CPU-DOUBLE-GFLOPS/second 0.350295

CPU Matrix multiplication completed. Time to launch GPU kernel.
==14804== Some kernel(s) will be replayed on device 0 in order to collect all events/metrics.
==14804== Replaying kernel "matmul_kernel(long, double*, double*, double*)" (done)

GPU took 9.038890 seconds as computed by CudaEvent function
GPU-DOUBLE-GFLOPS/second 5.974185
        fb_subp1_write_sectors
Experiment Done.
-------------------------------
==14804== Profiling application: MatMulDblGpuWin.exe 3000
==14804== Profiling result:
==14804== Metric result:
Invocations                               Metric Name                            Metric Description         Min         Max         Avg
Device "GeForce GTX 1050 (0)"
    Kernel: matmul_kernel(long, double*, double*, double*)
          1                             flop_count_dp   Floating Point Operations(Double Precision)  5.4000e+10  5.4000e+10  5.4000e+10
          1                    dram_read_transactions               Device Memory Read Transactions   758691589   758691589   758691589
          1                   dram_write_transactions              Device Memory Write Transactions     2572521     2572521     2572521

c:\dellmatmul\cuda10-2-2
```
Use NVPROF to gather CPU to GPU data movement and kernel statistics
```
c:\dellmatmul\cuda10-2-2>nvprof --print-gpu-trace MatMulDblGpuWin.exe 3000
==15580== NVPROF is profiling process 15580, command: MatMulDblGpuWin.exe 3000
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
CPU took 154.265000 seconds as computed by gettickcount
CPU-DOUBLE-GFLOPS/second 0.350047

CPU Matrix multiplication completed. Time to launch GPU kernel.

GPU took 0.951331 seconds as computed by CudaEvent function
GPU-DOUBLE-GFLOPS/second 56.762563

Experiment Done.
-------------------------------
==15580== Profiling application: MatMulDblGpuWin.exe 3000
==15580== Profiling result:
   Start  Duration            Grid Size      Block Size     Regs*    SSMem*    DSMem*      Size  Throughput  SrcMemType  DstMemType           Device   Context    Stream  Name
154.623s  5.5340ms                    -               -         -         -         -  68.665MB  12.117GB/s      Pinned      Device  GeForce GTX 105         1         7  [CUDA memcpy HtoD]
154.629s  5.4815ms                    -               -         -         -         -  68.665MB  12.233GB/s      Pinned      Device  GeForce GTX 105         1         7  [CUDA memcpy HtoD]
154.634s  934.47ms          (188 188 1)       (16 16 1)        32        0B        0B         -           -           -           -  GeForce GTX 105         1         7  matmul_kernel(long, double*, double*, double*) [120]
155.569s  5.6524ms                    -               -         -         -         -  68.665MB  11.863GB/s      Device      Pinned  GeForce GTX 105         1         7  [CUDA memcpy DtoH]

Regs: Number of registers used per CUDA thread. This number includes registers used internally by the CUDA driver and/or tools and can be more than what the compiler shows.
SSMem: Static shared memory allocated per CUDA block.
DSMem: Dynamic shared memory allocated per CUDA block.
SrcMemType: The type of source memory accessed by memory operation/copy
DstMemType: The type of destination memory accessed by memory operation/copy

c:\dellmatmul\cuda10-2-2>
```
## cuda linux I7 gtx1050
![results](http://davidjyoung.com/cmg/craydgemm.png)
## oneAPI windows I7 HD-630
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
## oneAPI linux I7 gtx1050
Compile
```
clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice matmul.dp.cpp -o matmul.dp.exe -std=c++17 -fsycl-unnamed-lambda


david@i77700:~/dellmatmul/oneapi/dpct_output$ clang++ -fsycl -fsycl-targets=nvptx64-nvidia-cuda-sycldevice matmul.dp.cpp -o matmul.dp.exe -std=c++17 -fsycl-unnamed-lambda
clang-13: warning: Unknown CUDA version. version.txt: 10.2.89. Assuming the latest supported version 10.1 [-Wunknown-cuda-version]
In file included from matmul.dp.cpp:2:
In file included from /home/david/includes/oneapi/dpct.hpp:17:
/home/david/includes/oneapi/device.hpp:203:9: warning: 'has_extension' is deprecated: use device::has() function with aspects APIs instead [-Wdeprecated-declarations]
    if (has_extension("cl_intel_required_subgroup_size")) {
        ^
/home/david/llvm/build/bin/../include/sycl/CL/sycl/device.hpp:164:3: note: 'has_extension' has been explicitly marked deprecated here
  __SYCL2020_DEPRECATED("use device::has() function with aspects APIs instead")
  ^
/home/david/llvm/build/bin/../include/sycl/CL/sycl/detail/defines_elementary.hpp:52:40: note: expanded from macro '__SYCL2020_DEPRECATED'
#define __SYCL2020_DEPRECATED(message) __SYCL_DEPRECATED(message)
                                       ^
/home/david/llvm/build/bin/../include/sycl/CL/sycl/detail/defines_elementary.hpp:43:38: note: expanded from macro '__SYCL_DEPRECATED'
#define __SYCL_DEPRECATED(message) [[deprecated(message)]]
                                     ^
matmul.dp.cpp:197:16: warning: format specifies type 'unsigned int' but the argument has type 'size_t' (aka 'unsigned long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
               ^~~~~~~~~~~~~~
matmul.dp.cpp:197:32: warning: format specifies type 'unsigned int' but the argument has type 'size_t' (aka 'unsigned long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
                               ^~~~~~~~~~~~~~
matmul.dp.cpp:197:48: warning: format specifies type 'unsigned int' but the argument has type 'size_t' (aka 'unsigned long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
                                               ^~~~~~~
matmul.dp.cpp:197:57: warning: format specifies type 'unsigned int' but the argument has type 'size_t' (aka 'unsigned long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
                                                        ^~~~~~~
5 warnings generated.
In file included from matmul.dp.cpp:2:
In file included from /home/david/includes/oneapi/dpct.hpp:17:
/home/david/includes/oneapi/device.hpp:203:9: warning: 'has_extension' is deprecated: use device::has() function with aspects APIs instead [-Wdeprecated-declarations]
    if (has_extension("cl_intel_required_subgroup_size")) {
        ^
/home/david/llvm/build/bin/../include/sycl/CL/sycl/device.hpp:164:5: note: 'has_extension' has been explicitly marked deprecated here
  [[deprecated("use device::has() function with aspects APIs instead")]]
    ^
matmul.dp.cpp:197:16: warning: format specifies type 'unsigned int' but the argument has type 'size_t' (aka 'unsigned long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
               ^~~~~~~~~~~~~~
matmul.dp.cpp:197:32: warning: format specifies type 'unsigned int' but the argument has type 'size_t' (aka 'unsigned long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
                               ^~~~~~~~~~~~~~
matmul.dp.cpp:197:48: warning: format specifies type 'unsigned int' but the argument has type 'size_t' (aka 'unsigned long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
                                               ^~~~~~~
matmul.dp.cpp:197:57: warning: format specifies type 'unsigned int' but the argument has type 'size_t' (aka 'unsigned long') [-Wformat]
               NUM_THREADS[2], NUM_THREADS[1], blks[2], blks[1]);
                                                        ^~~~~~~
5 warnings generated.
david@i77700:~/dellmatmul/oneapi/dpct_output$ ll
total 420
drwxrwx--- 2 david render   4096 Jul  6 03:28 ./
drwxrwxr-x 3 david david    4096 Jul  4 15:46 ../
-rwxrwxr-x 1 david render 169840 Jul  4 15:47 a.out*
-rw-rw-r-- 1 david render  37104 Jul  4 15:46 MainSourceFiles.yaml
-rw-rw-r-- 1 david render  10015 Jul  4 23:27 matmul.dp.cpp
-rwxrwxr-x 1 david david  192904 Jul  6 03:28 matmul.dp.exe*
david@i77700:~/dellmatmul/oneapi/dpct_output$
```
Execute
```
david@i77700:~/dellmatmul/oneapi/dpct_output$ SYCL_BE=PI_CUDA  ./matmul.dp.exe 1000

WARNING: The legacy environment variables SYCL_BE and SYCL_DEVICE_TYPE are deprecated. Please use SYCL_DEVICE_FILTER instead. For details, please refer to https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md

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

CPU took 3.328202 seconds as computed by gettimeofday() function
CPU-GDBLOPS/second 0.600925 

CPU Matrix multiplication completed. Time to launch GPU kernel.

GPU took 0.134232 seconds as computed by CudaEvent function
GPU-GDBLOPS/second 14.899591 

Experiment Done.
-------------------------------
david@i77700:~/dellmatmul/oneapi/dpct_output$
```
Platforms and Devices
```
david@i77700:~/oneapidiags$ SYCL_BE=PI_CUDA  ./test.dp.exe 

WARNING: The legacy environment variables SYCL_BE and SYCL_DEVICE_TYPE are deprecated. Please use SYCL_DEVICE_FILTER instead. For details, please refer to https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md

Platform: Intel(R) OpenCL HD Graphics
  Device: Intel(R) HD Graphics 630 [0x5912]
Platform: Intel(R) Level-Zero
  Device: Intel(R) HD Graphics 630 [0x5912]
Platform: NVIDIA CUDA BACKEND
  Device: GeForce GTX 1050
Platform: SYCL host platform
  Device: SYCL host device
0
2
4
6
8
10
12
14
16
18
david@i77700:~/oneapidiags$ 

```
Use ze_tracer to gather API statistics - https://github.com/intel/pti-gpu/tree/master/tools/ze_tracer
```
david@i77700:~/pti-gpu/tools/ze_tracer/build$ ./ze_tracer -c -h -v /home/david/dellmatmul/oneapi/dpct_output/a.out 3000
.
------ Matrix Dimensions ------
dims a,b = 3000 , 3000 
info: allocate host mem (205.99 MB)
.
info: device  mem (205.99 MB)
>>>> [329809155] zeMemAllocDevice: hContext = 0xc45c90 device_desc = 0x7ffcf0088d50 {UNKNOWN(0x0) 0 0 0} size = 72000000 alignment = 8 hDevice = 0xa1d4b0 pptr = 0x7ffcf0088da8 (ptr = 0)
<<<< [329847335] zeMemAllocDevice [38180 ns] ptr = 0xffffaaad55410000 -> ZE_RESULT_SUCCESS(0x0)
>>>> [329856451] zeMemAllocDevice: hContext = 0xc45c90 device_desc = 0x7ffcf0088d50 {UNKNOWN(0x0) 0 0 0} size = 72000000 alignment = 8 hDevice = 0xa1d4b0 pptr = 0x7ffcf0088da8 (ptr = 0)
<<<< [329877701] zeMemAllocDevice [21250 ns] ptr = 0xffffaaad598c0000 -> ZE_RESULT_SUCCESS(0x0)
>>>> [329883528] zeMemAllocDevice: hContext = 0xc45c90 device_desc = 0x7ffcf0088d50 {UNKNOWN(0x0) 0 0 0} size = 72000000 alignment = 8 hDevice = 0xa1d4b0 pptr = 0x7ffcf0088da8 (ptr = 0)
<<<< [329890890] zeMemAllocDevice [7362 ns] ptr = 0xffffaaad5dd70000 -> ZE_RESULT_SUCCESS(0x0)
Filling in 2D arrays a and b 
Filling Complete
------- CUDA Parameters -------
NUM_THREADS(  16,  16,   0)
       blks( 188, 188,   0)
TOTAL DBLOPS 54000000000.000000 
-------------------------------

Calling CPU Matrix Multiply 

CPU took 142.814397 seconds as computed by gettimeofday() function
CPU-GDBLOPS/second 0.378113 

CPU Matrix multiplication completed. Time to launch GPU kernel.
.
>>>> [143156646536] zeCommandListAppendMemoryCopy: hCommandList = 0xcdd720 dstptr = 0xffffaaad55410000 srcptr = 0x7fb8f84bf000 size = 72000000 hSignalEvent = 0x2186110 numWaitEvents = 0 phWaitEvents = 0
<<<< [143156771801] zeCommandListAppendMemoryCopy [125265 ns] -> ZE_RESULT_SUCCESS(0x0)
.
>>>> [143174946258] zeCommandListAppendMemoryCopy: hCommandList = 0xcdd720 dstptr = 0xffffaaad598c0000 srcptr = 0x7fb8f400d000 size = 72000000 hSignalEvent = 0x2171be0 numWaitEvents = 0 phWaitEvents = 0
<<<< [143174963119] zeCommandListAppendMemoryCopy [16861 ns] -> ZE_RESULT_SUCCESS(0x0)
.
>>>> [143300339680] zeCommandListAppendMemoryCopy: hCommandList = 0x2e7ddf0 dstptr = 0x7fb8f400d000 srcptr = 0xffffaaad5dd70000 size = 72000000 hSignalEvent = 0x2b01c90 numWaitEvents = 0 phWaitEvents = 0
<<<< [143300353749] zeCommandListAppendMemoryCopy [14069 ns] -> ZE_RESULT_SUCCESS(0x0)
.
GPU took 4.681859 seconds as computed by CudaEvent function
GPU-GDBLOPS/second 11.533880 

Experiment Done.
-------------------------------
=== API Timing Results: ===

Total Execution Time (ns):         147840060741
      Total API Time (ns):           4710111445

                              Function,       Calls,           Time (ns),  Time (%),        Average (ns),            Min (ns),            Max (ns)
                zeEventHostSynchronize,           4,          4551000472,     96.62,          1137750118,                3790,          4537845443
                        zeModuleCreate,           1,            99171288,      2.11,            99171288,            99171288,            99171288
     zeCommandQueueExecuteCommandLists,           4,            30621716,      0.65,             7655429,               33195,            10319992
                        zeMemAllocHost,           3,            28266066,      0.60,             9422022,             9283517,             9618434
          zeCommandListCreateImmediate,           1,              181099,      0.00,              181099,              181099,              181099
         zeCommandListAppendMemoryCopy,           3,              156195,      0.00,               52065,               14069,              125265
                  zeCommandQueueCreate,           1,              139411,      0.00,              139411,              139411,              139411
                   zeCommandListCreate,           2,               91209,      0.00,               45604,               43169,               48040
                      zeMemAllocDevice,           3,               66792,      0.00,               22264,                7362,               38180
                                zeInit,           1,               33355,      0.00,               33355,               33355,               33355
           zeDeviceGetMemoryProperties,          10,               33234,      0.00,                3323,                2983,                4230
                             zeMemFree,           3,               32988,      0.00,               10996,                5992,               20319
                    zeCommandListReset,           4,               32336,      0.00,                8084,                4063,               11450
            zeDeviceGetCacheProperties,          10,               30209,      0.00,                3020,                2827,                3755
                         zeEventCreate,           4,               28888,      0.00,                7222,                5661,                9020
                    zeFenceQueryStatus,           5,               18034,      0.00,                3606,                2744,                4171
              zeKernelSetArgumentValue,           4,               17280,      0.00,                4320,                3443,                6612
           zeDeviceGetModuleProperties,           5,               16976,      0.00,                3395,                2964,                4674
       zeCommandListAppendLaunchKernel,           1,               15822,      0.00,               15822,               15822,               15822
                    zeCommandListClose,           4,               15482,      0.00,                3870,                2818,                4407
                         zeFenceCreate,           2,               15138,      0.00,                7569,                7014,                8124
            zeDeviceGetImageProperties,           5,               13425,      0.00,                2685,                2515,                2782
               zeMemGetAllocProperties,           3,               11962,      0.00,                3987,                3074,                5689
                          zeFenceReset,           4,               10056,      0.00,                2514,                2317,                2693
                     zeEventPoolCreate,           1,                9749,      0.00,                9749,                9749,                9749
                       zeContextCreate,           1,                9127,      0.00,                9127,                9127,                9127
                        zeKernelCreate,           1,                8492,      0.00,                8492,                8492,                8492
zeDeviceGetCommandQueueGroupProperties,           2,                7969,      0.00,                3984,                3605,                4364
        zeDriverGetExtensionProperties,           2,                7520,      0.00,                3760,                3700,                3820
                           zeDeviceGet,           2,                7215,      0.00,                3607,                3119,                4096
                           zeDriverGet,           2,                6588,      0.00,                3294,                2797,                3791
                 zeDeviceGetSubDevices,           2,                5977,      0.00,                2988,                2640,                3337
                 zeDriverGetProperties,           1,                4506,      0.00,                4506,                4506,                4506
                  zeKernelSetGroupSize,           1,                4375,      0.00,                4375,                4375,                4375
       zeCommandListAppendWaitOnEvents,           1,                4101,      0.00,                4101,                4101,                4101
                 zeDeviceGetProperties,           1,                3778,      0.00,                3778,                3778,                3778
             zeKernelSetIndirectAccess,           1,                3680,      0.00,                3680,                3680,                3680
                 zeKernelGetProperties,           1,                3610,      0.00,                3610,                3610,                3610
          zeDeviceGetComputeProperties,           1,                2773,      0.00,                2773,                2773,                2773
                 zeDriverGetApiVersion,           1,                2552,      0.00,                2552,                2552,                2552

david@i77700:~/pti-gpu/tools/ze_tracer/build$ == API Timing Results: ===

```
