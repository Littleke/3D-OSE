#include <time.h>
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#define pai 3.14159265359

//CUFFT Header file
#include <cufftXt.h>


// includes, project
#include <cufft.h>
//#include <cutil_inline.h>
//#include <shrQATest.h>
#include "su.h"
#include "segy.h"
//#include "kexuan2.h"
#include "Complex.h"
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <timer.h>
#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

/*
	This is an seismic data modeling program.
	V1.0: 3D modeling on OSE method with multi-GPU;

		--Little Ke in Story
		2019.12.20
*/



// Complex data type
typedef float2 Complex;
#define batch 1

int nGPUs,M;
int nx,ny,nz,nt,bu,bd,bl,br,bf,bb,nxb,nyb,nzb,sxbeg,sybeg,szbeg,jsx,jsy,jsz,gxbeg,gybeg,gzbeg,jgx,jgy,jgz,csdgather,ng,nshot,is,it,sx,sy,sz,boundary;
float fm,dt,amp,dx,dy,dz,vmax,R,dtR,dkx,dky,dkz;

int *sxyz_h, *gxyz_h,*sxyz_d, *gxyz_d;

float *wavelet_h,*wavelet_d,*wavelethilbert_d;
cufftComplex *cwavelet_d;

float *wavelet_card,*wavelethilbert_card;
cufftComplex *cwavelet_card;


float *fdata_3dH,*fdata_3dD,*vb_3dH;
cufftComplex *cdata_3dH,*cvb_3dH;

cudaLibXtDesc *p1D,*p2D,*q1D,*q2D,*pcD,*vbD;
float *att_blr,*att_bfb,*att_bud;
float *att_host;

float *rec_slice_d,*rec_slice_h,*rec_data_h;


char parname[500];
char filename[500];
#include "ose_3d_kernel.cu"
#include "ose_3d_lib.cu"

int main()
{

	system("rm *.bin");
	system("rm ./wf/*.bin");
	nx=101;
	ny=101;
	nz=101;
	boundary=25;
	bu=boundary;
	bd=boundary;
	bl=boundary;
	br=boundary;
	bf=boundary;
	bb=boundary;
	nxb=bl+nx+br;
	nyb=bf+ny+bb;
	nzb=bu+nz+bd;
	nxb=judge_odd(&br,&nxb);
	nyb=judge_odd(&bb,&nyb);
	nzb=judge_odd(&bd,&nzb);

	printf("nxbnew=%d	brnew=%d\n",nxb,br);


	nt=300;
	fm=40.0;
	dt=1.0;
	dt=dt*0.001;
	amp=100.0;
	ng=nx*ny;
	nshot=1;
	sxbeg=nx/2;
	sybeg=ny/2;
	szbeg=0;
	gxbeg=0;
	gybeg=0;
	gzbeg=0;
	jsx=1;
	jsy=1;
	jsz=1;
	jgx=1;
	jgy=1;
	jgz=1;
	csdgather=0;
	nGPUs=2;
	dx=10.0;
	dy=10.0;
	dz=10.0;

	dkx=2*pai/dx/(float)nxb;
	dky=2*pai/dy/(float)nyb;
	dkz=2*pai/dz/(float)nzb;

	
	printf("dkx=%f	dky=%f	dkz=%f\n",dkx,dky,dkz);

	sprintf(parname,"./output/partest.txt");		// set the filename of orignal rtm result 

	cudaSetDevice(0);
	float mstimer;
	cudaEvent_t start, stop;
  	cudaEventCreate(&start);	
	cudaEventCreate(&stop);
	cudaEventRecord(start);/* record starting time */


	dim3 dimBlock2D(BLOCK_SIZE_X,BLOCK_SIZE_Y);
	dim3 dimGrid3D((nxb+dimBlock2D.x-1)/dimBlock2D.x,(nyb+dimBlock2D.y-1)/dimBlock2D.y,nzb);
	dim3 dimBlock_nt(BLOCK_SIZE_X,1);
	dim3 dimGrid_nt((nt+dimBlock_nt.x-1)/dimBlock_nt.x,1);

	dim3 dimGrid3D_half_before_tran((nxb+1+dimBlock2D.x-1)/dimBlock2D.x,(nyb+dimBlock2D.y-1)/dimBlock2D.y,nzb/2);
	dim3 dimGrid3D_half_after_tran((nxb+1+dimBlock2D.x-1)/dimBlock2D.x,(nyb/2+dimBlock2D.y-1)/dimBlock2D.y,nzb);

	Alloc();

	sg_init_host(sxyz_h,sxbeg,sybeg,szbeg,jsx,jsy,jsz,nshot,nx,ny);
	sg_init_host(gxyz_h,sxbeg,sybeg,szbeg,jsx,jsy,jsz,nshot,nx,ny);

	sg_init_device<<<1,1>>>(gxyz_d,gxbeg,gybeg,gzbeg,jgx,jgy,jgz,ng,nx,ny); //There is a potential BUG here, when receiver is put on non GPU0


//	init_fdata_3D(vb_3dH,nxb,nyb,nzb,3500.0);
//	init_cdata_r_3D(cvb_3dH,nxb,nyb,nzb,3500.0);


	sprintf(filename,"./input/v3d_nx101_ny101_nz101.bin");
	input_file_xyz_boundary(filename,vb_3dH,nx,ny,nz,bl,bf,bu,nxb,nyb,nzb);
	add_pml_layers_v_h(vb_3dH,nx,ny,nz,bl,bf,bu,nxb,nyb,nzb);
	data_R2CR(vb_3dH,cvb_3dH,nxb,nyb,nzb);


	Alloc_mulGPU();

	generate_wavelet_gpu<<<1,1>>>(wavelet_d,fm,nt,dt,amp);
	cudaMemcpy(wavelet_h,wavelet_d,nt*sizeof(float),cudaMemcpyDefault);
	write_1dfile_from_1darray("./temp/ricker.bin",wavelet_h,nt);

	cufftHandle plan_1d;
	cufftHandle plan_1d_r;

	checkCudaErrors(cufftPlan1d(&plan_1d,nt,CUFFT_R2C,1));
	checkCudaErrors(cufftPlan1d(&plan_1d_r,nt,CUFFT_C2R,1));


	checkCudaErrors(cufftExecR2C(plan_1d,wavelet_d,cwavelet_d));
	hilbert_1d<<<dimGrid_nt,dimBlock_nt>>>(cwavelet_d,nt);
	checkCudaErrors(cufftExecC2R(plan_1d_r,cwavelet_d,wavelethilbert_d));
	scale_wavelet<<<dimGrid_nt,dimBlock_nt>>>(wavelethilbert_d,nt);

	cudaMemcpy(wavelet_h,wavelethilbert_d,nt*sizeof(float),cudaMemcpyDefault);
	write_1dfile_from_1darray("./temp/ricker_hil.bin",wavelet_h,nt);


	for(int i=0;i<nt;i++)
	{
		printf("rickerhilbert[%d]=%f\n",i,wavelet_h[i]);
	}


	vmax=find_max(vb_3dH,nxb*nyb*nzb);
	printf("vmax=%f\n",vmax);

	R=vmax*PI*sqrt(pow(1.0/dx,2.0)+pow(1.0/dy,2.0)+pow(1.0/dz,2.0));
	printf("R=%f \n",R);
	dtR=dt*R;
	M=(int)dtR+6;

	printf("dtR=%f M=%d\n",dtR,M);


	R2C_3D_CPU(cdata_3dH,fdata_3dH,nxb,nyb,nzb,TRUE);

// Demonstrate how to use CUFFT to perform 3-d FFTs using 2 GPUs

// cufftCreate() - Create an empty plan
	cufftHandle plan_input;
	cufftResult result;

	result = cufftCreate(&plan_input);
	if (result != CUFFT_SUCCESS) { printf ("*Create failed\n"); return; }

// cufftXtSetGPUs() - Define which GPUs to use
	int whichGPUs[nGPUs],iGPU;
	for(iGPU=0;iGPU<nGPUs;iGPU++)
	{
		whichGPUs[iGPU] = iGPU;
	}
	result = cufftXtSetGPUs (plan_input, nGPUs, whichGPUs);
	if (result != CUFFT_SUCCESS) { printf ("*XtSetGPUs failed\n"); return; }

// Initialize FFT input data
	size_t worksize[nGPUs];

// cufftMakePlan3d() - Create the plan
	result = cufftMakePlan3d (plan_input, nzb, nyb, nxb, CUFFT_C2C, worksize);
	if (result != CUFFT_SUCCESS) { printf ("*MakePlan* failed\n"); return; }

// cufftXtMalloc() - Malloc data on multiple GPUs
	Alloc_cufftXt(plan_input,result);

// cufftXtMemcpy() - Copy data from host to multiple GPUs
	result = cufftXtMemcpy (plan_input, p1D,cdata_3dH, CUFFT_COPY_HOST_TO_DEVICE);
	if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed\n"); return; }

	result = cufftXtMemcpy (plan_input, vbD,cvb_3dH, CUFFT_COPY_HOST_TO_DEVICE);
	if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed\n"); return; }

// cufftXtExecDescriptorC2C() - Execute FFT on multiple GPUs


	for(is=0;is<nshot;is++)
	{
//		Pad Zero Values For Wavefield Data
		Xtdata_value(p1D,p2D,q1D,q2D,pcD,nxb,nyb,nzb,0.0);
		Alloc_wavelet(is,plan_1d,plan_1d_r);

		for(it=0;it<nt;it++)
		{
//			if(it<80)
//			{
				Xt_add_wavelet_sxyz(p2D,wavelet_card,sxyz_h,is);
				Xt_add_wavelet_sxyz(q2D,wavelethilbert_card,sxyz_h,is);
//			}
//			if(it%50==0)
			printf("is=%d	it=%d/%d	time=%f(s)/%f(s)\n",is,it,nt,dt*it,dt*nt);

			wavefield_extro_ori2(plan_input,result);

/*
			sprintf(filename,"./wf/q-%d-single.bin",it);
			result = cufftXtMemcpy (plan_input, cdata_3dH,q2D, CUFFT_COPY_DEVICE_TO_HOST);
			if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed\n"); return; }
			write_3dfile_from_1darray_ci(filename,cdata_3dH,nxb,nyb,nzb);

			sprintf(filename,"./wf/p-%d-single.bin",it);
			result = cufftXtMemcpy (plan_input, cdata_3dH,p2D, CUFFT_COPY_DEVICE_TO_HOST);
			if (result != CUFFT_SUCCESS) { printf ("*XtMemcpy failed\n"); return; }
			write_3dfile_from_1darray_ci(filename,cdata_3dH,nxb,nyb,nzb);
*/
			cudaSetDevice(0);	//There is a potential BUG here, when receiver is put on non GPU0
			rec_shotdata_3d<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) q2D->descriptor->data[0],rec_slice_d,gxyz_d,nx,ny,nz,bl,bf,bu,nxb,nyb,nzb,ng);
			cudaMemcpy(rec_slice_h,rec_slice_d,ng*sizeof(float),cudaMemcpyDefault);
			rec_slice_to_data(rec_slice_h,rec_data_h,ng,it);
		}
			write_3dfile_from_1darray("./output/qshot2.bin",rec_data_h,ng,1,nt);

		Free_wavelet(is);
	}



	Output();


	Free_cufftXt(plan_input,result);

// cufftDestroy() - Destroy FFT plan
	result = cufftDestroy(plan_input);
	if (result != CUFFT_SUCCESS) { printf ("*Destroy failed: code\n"); return; }

	result = cufftDestroy(plan_1d);
	if (result != CUFFT_SUCCESS) { printf ("plan_1d Destroy failed: code\n"); return; }

	result = cufftDestroy(plan_1d_r);
	if (result != CUFFT_SUCCESS) { printf ("plan_1d_r Destroy failed: code\n"); return; }

	Free();

	cudaSetDevice(0);
	cudaEventRecord(stop);/* record ending time */
  	cudaEventSynchronize(stop);
  	cudaEventElapsedTime(&mstimer, start, stop);
 	printf("finished: %f (s)\n",mstimer*1e-3); 



	return (0);
}
