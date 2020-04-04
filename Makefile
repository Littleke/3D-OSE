INC=${CWPROOT}/include
LIK=${CWPROOT}/lib
LIB=-lsu -lpar -lcwp -lm -lcufft -lcudart -lcusparse
#INCLUDES = /usr/local/cuda-5.5/samples/common/inc
#INCLUDES2 = /usr/local/cuda-5.5/include
#INCLUDES3 = /u10/TZE/lk/NVIDIA_CUDA-5.5_Samples/NVIDIA_CUDA-5.5_Samples/common/inc


INCLUDES = /usr/local/cuda-8.0/samples/common/inc
INCLUDES2 = /usr/local/cuda-8.0/include
INCLUDES3 = /u10/TZE/lk/CUDA-8.0-SAMPLES/NVIDIA_CUDA-8.0_Samples/common/inc


ALL:ose_3d


ose_3d:ose_3d.o
	nvcc -o ose_3d -g ose_3d.o -lm -I$(INCLUDES) -I$(INCLUDES2) -I$(INCLUDES3) -O3 -I$(INC) -L$(LIK) $(LIB)

ose_3d.o:ose_3d.cu 
	nvcc -c ose_3d.cu -I$(INCLUDES) -I$(INCLUDES2) -I$(INCLUDES3) -O3 -I$(INC) -L$(LIK) $(LIB)

clean:
	rm *.o

#ose_3d:
#	nvcc -o ose_3d -c ose_3d.cu -lm -I$(INCLUDES)

