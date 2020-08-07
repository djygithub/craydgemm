# craydgemm c++/cuda/hip double-precision DGEMM heat/power/smoke test
**DGEMM Matrix Multiplication** based on a script from Dell via UC Berkeley

**Copy craydgemm.zip from email to test directory, unzip it.**

`abc\@djy:\~/craydgemm\$ unzip craydgemm.zip`

`Archive: craydgemm.zip`

`inflating: craydgemm/MatMulCompilePrep.sh`

`inflating: craydgemm/Matrix_Multiplication_GPU_DOUBLE_GIOPS_PINNED_TWO.cu`

`extracting: craydgemm/mmdblgpu00.loop.sh`

`extracting: craydgemm/mmdblgpu01.loop.sh`

`extracting: craydgemm/mmdblgpu02.loop.sh`

`extracting: craydgemm/mmdblgpu03.loop.sh`

`extracting: craydgemm/mmdblgpu04.loop.sh`

`extracting: craydgemm/mmdblgpu05.loop.sh`

`extracting: craydgemm/mmdblgpu06.loop.sh`

`extracting: craydgemm/mmdblgpu07.loop.sh`

`inflating: craydgemm/rocmsmi.bw.sh`

`inflating: craydgemm/rocmsmi.sh`

`inflating: craydgemm/test4.sh`

`inflating: craydgemm/test8.sh`

`abc\@djy:\~/craydgemm\$`

**Use chmod to mark MatMulCompilePrep.sh executable, it will hipify, compile,
then execute mmdblgpugiops 10 times varying the matrix size from 1,000 doubles
to 10,000 doubles as a warm up. This script also marks the \*.sh files
executable.**

`abc\@djy:\~/craydgemm\$ cd craydgemm`

`abc\@djy:\~/craydgemm/craydgemm\$ chmod +x ./MatMulCompilePrep.sh`

`abc\@djy:\~/craydgemm/craydgemm\$ ./MatMulCompilePrep.sh`

`\++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++`

`MULTIPLYING TWO SQUARE MATRICES OF SIZE 1000 FLOATS`

`\------ Matrix Dimensions ------`

`dims a,b = 1000 , 1000`

`info: allocate host mem ( 22.89 MB)`

`info: device mem ( 22.89 MB)`

`Filling in 2D arrays a and b`

`Filling Complete`

`\------- CUDA Parameters -------`

`NUM_THREADS( 16, 16, 0)`

`blks( 63, 63, 0)`

`TOTAL DBLOPS 2000000000.000000`

`\-------------------------------`

`Calling CPU Matrix Multiply`

`CPU took 0.000000 seconds as computed by gettimeofday() function`

`CPU Matrix multiplication completed. Time to launch GPU kernel.`

`GPU took 0.371508 seconds as computed by CudaEvent function`

`GPU-GDBLOPS/second 5.383459`

`Experiment Done.`

`\-------------------------------`

`============================================================================================================`

`\++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++`

`MULTIPLYING TWO SQUARE MATRICES OF SIZE 2000 FLOATS`

**There are eight scripts provided for exercising eight gpus (0, 1, 2, 3, 4, 5,
'6, 7), each looks like the following with a matrix size of 25000 doubles.**

abc\@djy:\~/craydgemm/craydgemm\$ cat mmdblgpu00.loop.sh

export HIP_VISIBLE_DEVICES=0

while true

do

date

./mmdblgpugiops 25000

done

abc\@djy:\~/craydgemm/craydgemm\$

**Start an eight GPU stress test by executing ./test8**

abc\@djy:\~/craydgemm/craydgemm\$ ./test8.sh

Wed Sep 25 07:37:43 PDT 2019

Wed Sep 25 07:37:43 PDT 2019

Wed Sep 25 07:37:43 PDT 2019

abc\@djy:\~/craydgemm/craydgemm\$

Wed Sep 25 07:37:43 PDT 2019

Wed Sep 25 07:37:43 PDT 2019

Wed Sep 25 07:37:43 PDT 2019

Wed Sep 25 07:37:43 PDT 2019

Wed Sep 25 07:37:43 PDT 2019

\------ Matrix Dimensions ------

dims a,b = 25000 , 25000

info: allocate host mem (14305.11 MB)

\------ Matrix Dimensions ------

dims a,b = 25000 , 25000

info: allocate host mem (14305.11 MB)

\------ Matrix Dimensions ------

dims a,b = 25000 , 25000

info: allocate host mem (14305.11 MB)

\------ Matrix Dimensions ------

dims a,b = 25000 , 25000

info: allocate host mem (14305.11 MB)

\------ Matrix Dimensions ------

dims a,b = 25000 , 25000

info: allocate host mem (14305.11 MB)

\------ Matrix Dimensions ------

dims a,b = 25000 , 25000

info: allocate host mem (14305.11 MB)

\------ Matrix Dimensions ------

dims a,b = 25000 , 25000

info: allocate host mem (14305.11 MB)

\------ Matrix Dimensions ------

dims a,b = 25000 , 25000

info: allocate host mem (14305.11 MB)

**Start a rocmsmi.sh loop in another window to view GPU activity**

abc\@djy:\~/craydgemm/craydgemm\$ ./rocmsmi.sh

Wed Sep 25 07:37:25 PDT 2019

========================ROCm System Management Interface========================

================================================================================

GPU Temp AvgPwr SCLK MCLK Fan Perf PwrCap VRAM% GPU%

1 41.0c N/A 1606Mhz 1000Mhz 0.0% auto 225.0W 88% 100%

2 41.0c N/A 1606Mhz 1000Mhz 16.86% auto 225.0W 88% 100%

3 38.0c N/A 1725Mhz 1000Mhz 5.88% auto 225.0W 88% 100%

4 38.0c N/A 1725Mhz 1000Mhz 11.76% auto 225.0W 88% 100%

5 41.0c N/A 1725Mhz 1000Mhz 4.71% auto 225.0W 88% 100%

6 42.0c N/A 1725Mhz 1000Mhz 0.0% auto 225.0W 88% 100%

7 37.0c N/A 1606Mhz 1000Mhz 17.65% auto 225.0W 88% 100%

8 42.0c N/A 1606Mhz 1000Mhz 18.82% auto 225.0W 88% 100%

================================================================================

==============================End of ROCm SMI Log ==============================

**Use kill -9 -1 to kill the MatMul processes and all the tasks you can for your
userid. This will also kill any terminal sessions for your userid.**
