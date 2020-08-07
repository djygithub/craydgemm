/opt/rocm/hip/bin/hipify-perl Matrix_Multiplication_GPU_DOUBLE_GIOPS_PINNED_TWO.cu > mmgpudoublegiops.cpp

/opt/rocm/hip/bin/hipcc -O2 -o mmdblgpugiops  mmgpudoublegiops.cpp 

GB[1]=1000
GB[2]=2000
GB[3]=3000
GB[4]=4000
GB[5]=5000
GB[6]=6000
GB[7]=7000
GB[8]=8000
GB[9]=9000
GB[10]=10000






N=1
while [ $N -lt 11 ];
do
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "MULTIPLYING TWO SQUARE MATRICES OF SIZE  ${GB[$N]} FLOATS"

#./MatMul ${GB[$N]}

filename="pinned_${N}_GB.atp"
 ./mmdblgpugiops ${GB[$N]}


echo "============================================================================================================"

let N=N+1

done
chmod +x ./mmdblgpu00.loop.sh
chmod +x ./mmdblgpu01.loop.sh
chmod +x ./mmdblgpu02.loop.sh
chmod +x ./mmdblgpu03.loop.sh
chmod +x ./mmdblgpu04.loop.sh
chmod +x ./mmdblgpu05.loop.sh
chmod +x ./mmdblgpu06.loop.sh
chmod +x ./mmdblgpu07.loop.sh
chmod +x ./test4.sh
chmod +x ./test8.sh
chmod +x ./rocmsmi.sh
chmod +x ./rocmsmi.bw.sh

