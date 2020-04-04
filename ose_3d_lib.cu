
void write_1dfile_from_1darray(char *filename,float *Array1d,int nx)
{	
	FILE *fp;
	int ix,iz;
	if((fp=fopen(filename,"wb+"))==NULL)
	{	
		printf("Can not write this file : %s\n",filename);
		exit(0);
	}
	{
		for(ix=0;ix<nx;ix++)
			fwrite(&Array1d[ix],4,1,fp);
		fclose(fp);
	}
}


void read_1dfile_from_1darray(char *filename,float *Array1d,int nx)
{	
	FILE *fp;
	int ix,iz;
	if((fp=fopen(filename,"rb"))==NULL)
	{	
		printf("Can not read this file : %s\n",filename);
		exit(0);
	}
	{
		for(ix=0;ix<nx;ix++)
			fread(&Array1d[ix],4,1,fp);
		fclose(fp);
	}
}


int judge_odd(int *after,int *nb)
{
	if(*nb%2!=0)
	{
		*after=*after+1;
		*nb=*nb+1;
	}
	return (*nb);
}


void Alloc()
{

	checkCudaErrors(cudaMallocHost((void **)&sxyz_h,nshot*sizeof(int)));
	checkCudaErrors(cudaMemset(sxyz_h,0,nshot*sizeof(int)));

	checkCudaErrors(cudaMallocHost((void **)&gxyz_h,ng*sizeof(int)));
	checkCudaErrors(cudaMemset(gxyz_h,0,ng*sizeof(int)));

	checkCudaErrors(cudaMallocHost((void **)&gxyz_d,ng*sizeof(int)));
	checkCudaErrors(cudaMemset(gxyz_d,0,ng*sizeof(int)));

	checkCudaErrors(cudaMallocHost((void **)&rec_data_h,ng*nt*sizeof(float)));
	checkCudaErrors(cudaMemset(rec_data_h,0,ng*nt*sizeof(float)));

	checkCudaErrors(cudaMalloc((void **)&rec_slice_d,ng*sizeof(float)));
	checkCudaErrors(cudaMemset(rec_slice_d,0,ng*sizeof(float)));

	checkCudaErrors(cudaMallocHost((void **)&rec_slice_h,ng*sizeof(float)));
	checkCudaErrors(cudaMemset(rec_slice_h,0,ng*sizeof(float)));

	checkCudaErrors(cudaMallocHost((void **)&fdata_3dH,nxb * nyb * nzb*sizeof(float)));
	checkCudaErrors(cudaMemset(fdata_3dH,0,nxb * nyb * nzb*sizeof(float)));
	if (fdata_3dH == NULL) { printf ("fdata_3dH malloc failed\n"); return; }

	checkCudaErrors(cudaMalloc((void **)&fdata_3dD,nxb * nyb * nzb*sizeof(float)));
	checkCudaErrors(cudaMemset(fdata_3dD,0,nxb * nyb * nzb*sizeof(float)));
	if (fdata_3dD == NULL) { printf ("fdata_3dD malloc failed\n"); return; }

	checkCudaErrors(cudaMallocHost((void **)&vb_3dH,nxb * nyb * nzb*sizeof(float)));
	checkCudaErrors(cudaMemset(vb_3dH,0,nxb * nyb * nzb*sizeof(float)));
	if (fdata_3dD == NULL) { printf ("vb_3dH malloc failed\n"); return; }

	checkCudaErrors(cudaMallocHost((void **)&cvb_3dH,nxb * nyb * nzb*sizeof(cufftComplex)));
	checkCudaErrors(cudaMemset(cvb_3dH,0,nxb * nyb * nzb*sizeof(float)));
	if (fdata_3dD == NULL) { printf ("cvb_3dH malloc failed\n"); return; }



	checkCudaErrors(cudaMallocHost((void **)&cdata_3dH,nxb * nyb * nzb*sizeof(cufftComplex)));
	checkCudaErrors(cudaMemset(cdata_3dH,0,nxb * nyb * nzb*sizeof(cufftComplex)));

	checkCudaErrors(cudaMallocHost((void **)&wavelet_h,nt*sizeof(float)));
	checkCudaErrors(cudaMemset(wavelet_h,0,nt*sizeof(float)));

	checkCudaErrors(cudaMalloc((void **)&wavelet_d,nt*sizeof(float)));
	checkCudaErrors(cudaMemset(wavelet_d,0,nt*sizeof(float)));

	checkCudaErrors(cudaMalloc((void **)&wavelethilbert_d,nt*sizeof(float)));
	checkCudaErrors(cudaMemset(wavelethilbert_d,0,nt*sizeof(float)));

	checkCudaErrors(cudaMalloc((void **)&cwavelet_d,nt*sizeof(cufftComplex)));
	checkCudaErrors(cudaMemset(cwavelet_d,0,nt*sizeof(cufftComplex)));

	checkCudaErrors(cudaMallocHost((void **)&att_host,300*sizeof(float)));
	checkCudaErrors(cudaMemset(att_host,0,300*sizeof(float)));


}


void synchronize_gpus(int nGPUs)
{
	int device;
	// Wait for device to finish all operation
	for( int i=0; i< nGPUs ; i++ )
	{
		checkCudaErrors(cudaSetDevice(i));
		cudaDeviceSynchronize();
		// Check if kernel execution generated and error
		getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");
	}

}


void Alloc_mulGPU()
{
	for(int i=0; i < nGPUs ;i++)
	{
//		device = data->descriptor->GPUs[i];
		//Set device

		checkCudaErrors(cudaSetDevice(i));
		checkCudaErrors(cudaMalloc((void **)&att_blr,nxb*sizeof(float)));
		checkCudaErrors(cudaMemset(att_blr,0,nxb*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&att_bfb,nyb*sizeof(float)));
		checkCudaErrors(cudaMemset(att_bfb,0,nyb*sizeof(float)));

		checkCudaErrors(cudaMalloc((void **)&att_bud,nzb*sizeof(float)));
		checkCudaErrors(cudaMemset(att_bud,0,nzb*sizeof(float)));

		generate_att<<<1,1>>>(att_blr,bl,nx,br);
		generate_att<<<1,1>>>(att_bfb,bf,ny,bb);
		generate_att<<<1,1>>>(att_bud,bu,nz,bd);

	}

	synchronize_gpus(nGPUs);


	checkCudaErrors(cudaSetDevice(1));
	cudaMemcpy(att_host,att_blr,nxb*sizeof(float),cudaMemcpyDefault);
	write_1dfile_from_1darray("./temp/att_x_1.bin",att_host,300);



	checkCudaErrors(cudaSetDevice(0));
	cudaMemcpy(att_host,att_blr,nxb*sizeof(float),cudaMemcpyDefault);
	write_1dfile_from_1darray("./temp/att_x_0.bin",att_host,300);

}


void Alloc_cufftXt(cufftHandle plan_input,cufftResult result)
{
// cufftXtMalloc() - Malloc data on multiple GPUs
//	cudaLibXtDesc *p1D;
	result = cufftXtMalloc (plan_input, &p1D,CUFFT_XT_FORMAT_INPLACE);
	if (result != CUFFT_SUCCESS) { printf ("p1D XtMalloc failed\n"); return; }

	result = cufftXtMalloc (plan_input, &p2D,CUFFT_XT_FORMAT_INPLACE);
	if (result != CUFFT_SUCCESS) { printf ("p2D XtMalloc failed\n"); return; }

	result = cufftXtMalloc (plan_input, &q1D,CUFFT_XT_FORMAT_INPLACE);
	if (result != CUFFT_SUCCESS) { printf ("q1D XtMalloc failed\n"); return; }

	result = cufftXtMalloc (plan_input, &q2D,CUFFT_XT_FORMAT_INPLACE);
	if (result != CUFFT_SUCCESS) { printf ("q2D XtMalloc failed\n"); return; }

	result = cufftXtMalloc (plan_input, &pcD,CUFFT_XT_FORMAT_INPLACE);
	if (result != CUFFT_SUCCESS) { printf ("pcD XtMalloc failed\n"); return; }

	result = cufftXtMalloc (plan_input, &vbD,CUFFT_XT_FORMAT_INPLACE);
	if (result != CUFFT_SUCCESS) { printf ("vbD XtMalloc failed\n"); return; }

}

void Free_cufftXt(cufftHandle plan_input,cufftResult result)
{

// cufftXtFree() - Free GPU memory
	result = cufftXtFree(p1D);
	if (result != CUFFT_SUCCESS) { printf ("p1D XtFree failed\n"); return; }

	result = cufftXtFree(p2D);
	if (result != CUFFT_SUCCESS) { printf ("p2D XtFree failed\n"); return; }

	result = cufftXtFree(q1D);
	if (result != CUFFT_SUCCESS) { printf ("q1D XtFree failed\n"); return; }

	result = cufftXtFree(q2D);
	if (result != CUFFT_SUCCESS) { printf ("q2D XtFree failed\n"); return; }

	result = cufftXtFree(pcD);
	if (result != CUFFT_SUCCESS) { printf ("pcD XtFree failed\n"); return; }
	
	result = cufftXtFree(vbD);
	if (result != CUFFT_SUCCESS) { printf ("vbD XtFree failed\n"); return; }
}



void Free_mulGPU()
{
	for(int i=0; i < nGPUs ;i++)
	{
//		device = data->descriptor->GPUs[i];
		//Set device
		cudaFree(att_blr);
		cudaFree(att_bfb);
		cudaFree(att_bud);
	}
	synchronize_gpus(nGPUs);

}

void Free()
{


	cudaFree(sxyz_h);
	cudaFree(gxyz_h);	
	cudaFree(gxyz_d);	
	cudaFree(rec_slice_d);
	cudaFree(rec_slice_h);
	cudaFree(rec_data_h);

	cudaFree(vb_3dH);
	cudaFree(cvb_3dH);

	cudaFree(fdata_3dH);
	cudaFree(fdata_3dD);
	cudaFree(cdata_3dH);
	cudaFree(wavelet_d);
	cudaFree(wavelet_h);
	cudaFree(wavelethilbert_d);
	cudaFree(cwavelet_d);

	cudaFree(att_host);

	
}




void output_par(char *parname)
{
	FILE *fp;
	if((fp=fopen(parname,"wb+"))==NULL)
	{	
		printf("Can not write this file : %s\n",parname);
		exit(0);
	}
	{
		{
			fprintf(fp,"X dimension size----> nx\n");
			fprintf(fp,"%d\n",nx);
			fprintf(fp,"Y dimension size----> ny\n");
			fprintf(fp,"%d\n",ny);
			fprintf(fp,"Z dimension size----> ny\n");
			fprintf(fp,"%d\n",nz);
			fprintf(fp,"The interval of samples in space domain X dimension ----> dx (the units is 'mile')\n");
			fprintf(fp,"%f\n",dx);
			fprintf(fp,"The interval of samples in space domain Y dimension ----> dy (the units is 'mile')\n");
			fprintf(fp,"%f\n",dy);
			fprintf(fp,"The interval of samples in space domain Z dimension ----> dz (the units is 'mile')\n");
			fprintf(fp,"%f\n",dz);
			fprintf(fp,"The number of samples in time domain ----> nt\n");
			fprintf(fp,"%d\n",nt);
			fprintf(fp,"The number of receivers gather ----> ng\n");
			fprintf(fp,"%d\n",ng);
			fprintf(fp,"The number of sources ----> ns\n");
			fprintf(fp,"%d\n",nshot);
			fprintf(fp,"The interval of samples in time domain (the units is 's') ----> dt\n");
			fprintf(fp,"%f\n",dt);
			fprintf(fp,"The amplitude of wavelet ----> amp\n");
			fprintf(fp,"%f\n",amp);
			fprintf(fp,"The main frequency of wavelet ----> fm\n");
			fprintf(fp,"%f\n",fm);
			fprintf(fp,"The beginning location of sources in X-axis ----> sxbeg\n");
			fprintf(fp,"%d\n",sxbeg);
			fprintf(fp,"The beginning location of sources in Y-axis ----> sxbeg\n");
			fprintf(fp,"%d\n",sybeg);
			fprintf(fp,"The beginning location of sources in Z-axis ----> szbeg\n");
			fprintf(fp,"%d\n",szbeg);
			fprintf(fp,"The beginning location of receivers in X-axis ----> gxbeg\n");
			fprintf(fp,"%d\n",gxbeg);
			fprintf(fp,"The beginning location of receivers in Y-axis ----> gybeg\n");
			fprintf(fp,"%d\n",gybeg);
			fprintf(fp,"The beginning location of receivers in Z-axis ----> gzbeg\n");
			fprintf(fp,"%d\n",gzbeg);
			fprintf(fp,"Sources X-axis  jump interval ----> jsx\n");
			fprintf(fp,"%d\n",jsx);
			fprintf(fp,"Sources Y-axis  jump interval ----> jsy\n");
			fprintf(fp,"%d\n",jsy);
			fprintf(fp,"Sources Z-axis  jump interval ----> jsz\n");
			fprintf(fp,"%d\n",jsz);
			fprintf(fp,"Receivers X-axis  jump interval ----> jgx\n");
			fprintf(fp,"%d\n",jgx);
			fprintf(fp,"Receivers Y-axis  jump interval ----> jgy\n");
			fprintf(fp,"%d\n",jgy);
			fprintf(fp,"Receivers Z-axis  jump interval ----> jgzn");
			fprintf(fp,"%d\n",jgz);
			fprintf(fp,"Default, common shot-gather; if n, record at every point\n");
			fprintf(fp,"%d\n",csdgather);
		}
		fclose(fp);
	}

}

void Output()
{
	output_par(parname);

}

void write_3dfile_from_1darray(char *filename,float *Array1d,int nx,int ny,int nz)
{	
	FILE *fp;
	int ix,iy,iz;
	if((fp=fopen(filename,"wb+"))==NULL)
	{	
		printf("Can not write this file : %s\n",filename);
		exit(0);
	}
	{
		for(iy=0;iy<ny;iy++)
		for(ix=0;ix<nx;ix++)
		for(iz=0;iz<nz;iz++)	
			fwrite(&Array1d[iz*nx*ny+iy*nx+ix],4,1,fp);
	}
	fclose(fp);
}

void write_3dfile_from_1darray_cr(char *filename,cufftComplex *Array1d,int nx,int ny,int nz)
{	
	FILE *fp;
	int ix,iy,iz;
	if((fp=fopen(filename,"wb+"))==NULL)
	{	
		printf("Can not write this file : %s\n",filename);
		exit(0);
	}
	{
		for(iy=0;iy<ny;iy++)
		for(ix=0;ix<nx;ix++)
		for(iz=0;iz<nz;iz++)	
			fwrite(&Array1d[iz*nx*ny+iy*nx+ix].x,4,1,fp);
	}
	fclose(fp);
}


void write_3dfile_from_1darray_ci(char *filename,cufftComplex *Array1d,int nx,int ny,int nz)
{	
	FILE *fp;
	int ix,iy,iz;
	if((fp=fopen(filename,"wb+"))==NULL)
	{	
		printf("Can not write this file : %s\n",filename);
		exit(0);
	}
	{
		for(iy=0;iy<ny;iy++)
		for(ix=0;ix<nx;ix++)
		for(iz=0;iz<nz;iz++)	
			fwrite(&Array1d[iz*nx*ny+iy*nx+ix].y,4,1,fp);
	}
	fclose(fp);
}


void R2C_3D_CPU(cufftComplex *complex,float *real,int nx,int ny,int nz,bool yes)
{
	int ix,iy,iz,id;
	for(iz=0;iz<nz;iz++)
	for(iy=0;iy<ny;iy++)
	for(ix=0;ix<nx;ix++)
	{
		id=iz*ny*nx+iy*nx+ix;
		if(yes)
		{
			complex[id].x=real[id];
			complex[id].y=0.0;
		}	
		else
		{
			real[id]=complex[id].x;
		}

	}

}




void amp_3d_CPU(float *real,cufftComplex *complex,int nx,int ny,int nz)
{
	int ix,iy,iz,id;
	for(iz=0;iz<nz;iz++)
	for(iy=0;iy<ny;iy++)
	for(ix=0;ix<nx;ix++)
	{
		id=iz*ny*nx+iy*nx+ix;
		real[id]=sqrt(complex[id].x*complex[id].x+complex[id].y*complex[id].y);
//		real[id]=complex[id].x;
	}
}


void init_fdata_3D(float *data, int nx, int ny, int nz, float value)
{
	int ix,iy,iz,id;
	for(iz=0;iz<nz;iz++)
	for(iy=0;iy<ny;iy++)
	for(ix=0;ix<nx;ix++)
	{
		id=iz*ny*nx+iy*nx+ix;
		data[id]=value;
	}
}

void init_cdata_r_3D(cufftComplex *data, int nx, int ny, int nz, float value)
{
	int ix,iy,iz,id;
	for(iz=0;iz<nz;iz++)
	for(iy=0;iy<ny;iy++)
	for(ix=0;ix<nx;ix++)
	{
		id=iz*ny*nx+iy*nx+ix;
		data[id].x=value;
	}
}

void data_R2CR(float *rdata,cufftComplex *cdata, int nx, int ny, int nz)
{
	int ix,iy,iz,id;
	for(iz=0;iz<nz;iz++)
	for(iy=0;iy<ny;iy++)
	for(ix=0;ix<nx;ix++)
	{
		id=iz*ny*nx+iy*nx+ix;
		cdata[id].x=rdata[id];
	}


}

void Xtdata_value_before_fft(cudaLibXtDesc *data, int nxb, int nyb, int nzb, float value)
{
	int device;
	dim3 dimBlock2D(BLOCK_SIZE_X,BLOCK_SIZE_Y);
	dim3 dimGrid3D_half_before_tran((nxb+1+dimBlock2D.x-1)/dimBlock2D.x,(nyb+dimBlock2D.y-1)/dimBlock2D.y,nzb/2);


		for(int i=0; i < nGPUs ;i++)
		{
//			device = data->descriptor->GPUs[i];
			//Set device
			checkCudaErrors(cudaSetDevice(i));
			Xtdata_devide_value<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) data->descriptor->data[i],nxb,nyb,nzb/2,i,value);
		}
		// Wait for device to finish all operation
		for( int i=0; i< nGPUs ; i++ )
		{
//			device = data->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(i));
			cudaDeviceSynchronize();
			// Check if kernel execution generated and error
			getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");
		}
}

void Xtdata_value_after_fft(cudaLibXtDesc *data, int nxb, int nyb, int nzb, float value)
{
	int device;
	dim3 dimBlock2D(BLOCK_SIZE_X,BLOCK_SIZE_Y);
	dim3 dimGrid3D_half_after_tran((nxb+1+dimBlock2D.x-1)/dimBlock2D.x,(nyb/2+dimBlock2D.y-1)/dimBlock2D.y,nzb);

		for(int i=0; i < nGPUs ;i++)
		{
//			device = data->descriptor->GPUs[i];
			//Set device
			checkCudaErrors(cudaSetDevice(i));
			Xtdata_devide_value<<<dimGrid3D_half_after_tran,dimBlock2D>>>((cufftComplex*) data->descriptor->data[i],nxb,nyb/2,nzb,i,value);
		}
		// Wait for device to finish all operation
		for( int i=0; i< nGPUs ; i++ )
		{
//			device = data->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(i));
			cudaDeviceSynchronize();
			// Check if kernel execution generated and error
			getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");
		}
}





void Xtdata_value(cudaLibXtDesc *p1D, cudaLibXtDesc *p2D,cudaLibXtDesc *q1D,cudaLibXtDesc *q2D,cudaLibXtDesc *pcD, int nxb, int nyb, int nzb, float value)
{
	int device;
	dim3 dimBlock2D(BLOCK_SIZE_X,BLOCK_SIZE_Y);
	dim3 dimGrid3D_half_before_tran((nxb+1+dimBlock2D.x-1)/dimBlock2D.x,(nyb+dimBlock2D.y-1)/dimBlock2D.y,nzb/2);


		for(int i=0; i < nGPUs ;i++)
		{
			device = p1D->descriptor->GPUs[i];
			//Set device
			checkCudaErrors(cudaSetDevice(device));
			Xtdata_devide_value<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) p1D->descriptor->data[i],nxb,nyb,nzb/2,i,value);
			Xtdata_devide_value<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) p2D->descriptor->data[i],nxb,nyb,nzb/2,i,value);
			Xtdata_devide_value<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) q1D->descriptor->data[i],nxb,nyb,nzb/2,i,value);
			Xtdata_devide_value<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) q2D->descriptor->data[i],nxb,nyb,nzb/2,i,value);
			Xtdata_devide_value<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) pcD->descriptor->data[i],nxb,nyb,nzb/2,i,value);
		}
		// Wait for device to finish all operation
		for( int i=0; i< nGPUs ; i++ )
		{
			device = p1D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));
			cudaDeviceSynchronize();
			// Check if kernel execution generated and error
			getLastCudaError("Kernel execution failed [ ComplexPointwiseMulAndScale ]");
		}


}

float find_max(float *data,int size)
{
	float max=-9999999.999;
	int i;
	for(i=0;i<size;i++)
	{
		if(data[i]>max)
		max=data[i];
	}

	return max;
}

void Alloc_wavelet(int is,cufftHandle plan_1d,cufftHandle plan_1d_r)
{
	dim3 dimBlock_nt(BLOCK_SIZE_X,1);
	dim3 dimGrid_nt((nt+dimBlock_nt.x-1)/dimBlock_nt.x,1);

	int sz;
	sz=szbeg+is*jsz;	
	cudaSetDevice(sz/(nzb/2));
	checkCudaErrors(cudaMalloc((void **)&wavelet_card,nt*sizeof(float)));
	checkCudaErrors(cudaMemset(wavelet_card,0,nt*sizeof(float)));

	checkCudaErrors(cudaMalloc((void **)&wavelethilbert_card,nt*sizeof(float)));
	checkCudaErrors(cudaMemset(wavelethilbert_card,0,nt*sizeof(float)));

	checkCudaErrors(cudaMalloc((void **)&cwavelet_card,nt*sizeof(cufftComplex)));
	checkCudaErrors(cudaMemset(cwavelet_card,0,nt*sizeof(cufftComplex)));


	generate_wavelet_gpu<<<1,1>>>(wavelet_card,fm,nt,dt,amp);
	cudaMemcpy(wavelet_h,wavelet_card,nt*sizeof(float),cudaMemcpyDefault);
	write_1dfile_from_1darray("./temp/ricker_card.bin",wavelet_h,nt);
	checkCudaErrors(cufftExecR2C(plan_1d,wavelet_card,cwavelet_card));

	hilbert_1d<<<dimGrid_nt,dimBlock_nt>>>(cwavelet_card,nt);
	checkCudaErrors(cufftExecC2R(plan_1d_r,cwavelet_card,wavelethilbert_card));
	cudaMemcpy(wavelet_h,wavelethilbert_card,nt*sizeof(float),cudaMemcpyDefault);
	write_1dfile_from_1darray("./temp/hilbert_card.bin",wavelet_h,nt);


}

void Free_wavelet(int is)
{
	int sz;
	sz=szbeg+is*jsz;	
	cudaSetDevice(sz/(nzb/2));
	cudaFree(wavelet_card);
	cudaFree(wavelethilbert_card);
	cudaFree(cwavelet_card);
}

void sg_init_host(int *xyz_d, int xbeg, int ybeg, int zbeg, int jx, int jy, int jz, int number, int nx, int ny)
{
	int i,x,y,z;
	for(i=0;i<number;i++)
	{
		x=xbeg+i*jx;
		y=ybeg+i*jy;
		z=zbeg+i*jz;
		xyz_d[i]=z*nx*ny+y*nx+x;
	}
}

void Xt_add_wavelet_sxyz(cudaLibXtDesc *dataD,float *wavelet,int *sxyz_h,int is)
{
	int sx,sy,sz;
	sx=sxyz_h[is]%(nx*ny)%nx;
	sy=sxyz_h[is]%(nx*ny)/nx;
	sz=sxyz_h[is]/(nx*ny);
	if(sz<(int)(1.0*nz*0.5+0.5))
	{
		cudaSetDevice(0);
		add_wavelet_Xt<<<1,1>>>((cufftComplex*) dataD->descriptor->data[0],wavelet,nxb,nyb,nzb/2,bl,bf,bu,sx,sy,sz,it,0);
		if(it==0)
		printf("Wavelets are add on GPU-0	sx=%d	sy=%d	sz=%d\n",sx,sy,sz);
	}
	else
	{
		cudaSetDevice(1);
		add_wavelet_Xt<<<1,1>>>((cufftComplex*) dataD->descriptor->data[1],wavelet,nxb,nyb,nzb/2,bl,bf,bu,sx,sy,sz,it,1);
		if(it==1)
		printf("Wavelets are add on GPU-1	sx=%d	sy=%d	sz=%d\n",sx,sy,sz);
	}



}





void wavefield_extro_ori2(cufftHandle plan_input, cufftResult result)
{

	dim3 dimBlock2D(BLOCK_SIZE_X,BLOCK_SIZE_Y);
	dim3 dimGrid3D((nxb+dimBlock2D.x-1)/dimBlock2D.x,(nyb+dimBlock2D.y-1)/dimBlock2D.y,nzb);
	dim3 dimGrid3D_half_before_tran((nxb+1+dimBlock2D.x-1)/dimBlock2D.x,(nyb+dimBlock2D.y-1)/dimBlock2D.y,nzb/2);
	dim3 dimGrid3D_half_after_tran((nxb+1+dimBlock2D.x-1)/dimBlock2D.x,(nyb/2+dimBlock2D.y-1)/dimBlock2D.y,nzb);

	int device,n;
	//Launch the ComplexPointwiseMulAndScale<<< >>> kernel on multiple GPU
	Xtdata_value_before_fft(pcD,nxb,nyb,nzb,0.0);

	for(int i=0; i < nGPUs ;i++)
	{
			device = p1D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));

		//	STEP 1 - 4
		n=0;
		step1to4_3d<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) p2D->descriptor->data[i],nxb,nyb,nzb/2,dtR,n);
		step1to4_3d<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) q2D->descriptor->data[i],nxb,nyb,nzb/2,dtR,n);
		//	STEP 5
		n=1;
		memcpy_between_device_cr<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) q2D->descriptor->data[i],(cufftComplex*) pcD->descriptor->data[i],nxb,nyb,nzb/2);
	}

	synchronize_gpus(nGPUs);


	result = cufftXtExecDescriptorC2C (plan_input, pcD,pcD, CUFFT_FORWARD);
	if (result != CUFFT_SUCCESS) { printf ("pcD *XtExec* failed\n"); return; }

	for(int i=0; i < nGPUs ;i++)
	{
			device = p1D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));
		scale_kxkz_3D<<<dimGrid3D_half_after_tran,dimBlock2D>>>((cufftComplex*) pcD->descriptor->data[i],nxb,nyb/2,nzb,i,dkx,dky,dkz);
	}

	synchronize_gpus(nGPUs);

	result = cufftXtExecDescriptorC2C (plan_input, pcD,pcD, CUFFT_INVERSE);
	if (result != CUFFT_SUCCESS) { printf ("pcD INVERSE *XtExec* failed\n"); return; }

	for(int i=0; i < nGPUs ;i++)
	{
			device = p1D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));
		scal_vr_3d<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) pcD->descriptor->data[i],(cufftComplex*) p1D->descriptor->data[i],(cufftComplex*) vbD->descriptor->data[i],R,nxb,nyb,nzb/2,1.0,FALSE);


	}

	synchronize_gpus(nGPUs);


	//	STEP 6
	Xtdata_value_before_fft(pcD,nxb,nyb,nzb,0.0);

	for(int i=0; i < nGPUs ;i++)
	{
			device = p1D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));
		memcpy_between_device_cr<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) p2D->descriptor->data[i],(cufftComplex*) pcD->descriptor->data[i],nxb,nyb,nzb/2);
	}
	synchronize_gpus(nGPUs);



	result = cufftXtExecDescriptorC2C (plan_input, pcD,pcD, CUFFT_FORWARD);
	if (result != CUFFT_SUCCESS) { printf ("pcD *XtExec* failed\n"); return; }

	for(int i=0; i < nGPUs ;i++)
	{
			device = p1D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));
		scale_kxkz_3D<<<dimGrid3D_half_after_tran,dimBlock2D>>>((cufftComplex*) pcD->descriptor->data[i],nxb,nyb/2,nzb,i,dkx,dky,dkz);
	}

	synchronize_gpus(nGPUs);

	result = cufftXtExecDescriptorC2C (plan_input, pcD,pcD, CUFFT_INVERSE);
	if (result != CUFFT_SUCCESS) { printf ("pcD INVERSE *XtExec* failed\n"); return; }


	for(int i=0; i < nGPUs ;i++)
	{
			device = p1D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));
		scal_vr_3d<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) pcD->descriptor->data[i],(cufftComplex*) q1D->descriptor->data[i],(cufftComplex*) vbD->descriptor->data[i],R,nxb,nyb,nzb/2,-1.0,FALSE);


	}

	synchronize_gpus(nGPUs);


	for(int i=0; i < nGPUs ;i++)
	{
			device = p1D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));
		//	STEP 7
		add_bessel_3d<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) p1D->descriptor->data[i],(cufftComplex*) p2D->descriptor->data[i],dtR,n,nxb,nyb,nzb/2);
		//	STEP 8
		add_bessel_3d<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) q1D->descriptor->data[i],(cufftComplex*) q2D->descriptor->data[i],dtR,n,nxb,nyb,nzb/2);
	}

	synchronize_gpus(nGPUs);



	for(n=1;n<=M;n++)
	{
//		printf("it=%d	M=%d	n=%d\n",it,M,n);
		//	STEP 10
		Xtdata_value_before_fft(pcD,nxb,nyb,nzb,0.0);
		for(int i=0; i < nGPUs ;i++)
		{
			device = p1D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));
			memcpy_between_device_cr<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) p1D->descriptor->data[i],(cufftComplex*) pcD->descriptor->data[i],nxb,nyb,nzb/2);
		}

		synchronize_gpus(nGPUs);


		result = cufftXtExecDescriptorC2C (plan_input, pcD,pcD, CUFFT_FORWARD);
		if (result != CUFFT_SUCCESS) { printf ("pcD *XtExec* failed\n"); return; }


		for(int i=0; i < nGPUs ;i++)
		{
			device = p1D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));
			scale_kxkz_3D<<<dimGrid3D_half_after_tran,dimBlock2D>>>((cufftComplex*) pcD->descriptor->data[i],nxb,nyb/2,nzb,i,dkx,dky,dkz);
		}

		synchronize_gpus(nGPUs);

		result = cufftXtExecDescriptorC2C (plan_input, pcD,pcD, CUFFT_INVERSE);
		if (result != CUFFT_SUCCESS) { printf ("pcD INVERSE *XtExec* failed\n"); return; }



		for(int i=0; i < nGPUs ;i++)
		{
			device = p1D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));
			scal_vr_3d<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) pcD->descriptor->data[i],(cufftComplex*) q2D->descriptor->data[i],(cufftComplex*) vbD->descriptor->data[i],R,nxb,nyb,nzb/2,-2.0,TRUE);

		}

		synchronize_gpus(nGPUs);



		//	STEP 11
		Xtdata_value_before_fft(pcD,nxb,nyb,nzb,0.0);
		for(int i=0; i < nGPUs ;i++)
		{
			device = p1D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));
			memcpy_between_device_cr<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) q1D->descriptor->data[i],(cufftComplex*) pcD->descriptor->data[i],nxb,nyb,nzb/2);
		}

		synchronize_gpus(nGPUs);


		result = cufftXtExecDescriptorC2C (plan_input, pcD,pcD, CUFFT_FORWARD);
		if (result != CUFFT_SUCCESS) { printf ("pcD *XtExec* failed\n"); return; }


		for(int i=0; i < nGPUs ;i++)
		{
			device = p1D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));
			scale_kxkz_3D<<<dimGrid3D_half_after_tran,dimBlock2D>>>((cufftComplex*) pcD->descriptor->data[i],nxb,nyb/2,nzb,i,dkx,dky,dkz);
		}

		synchronize_gpus(nGPUs);

		result = cufftXtExecDescriptorC2C (plan_input, pcD,pcD, CUFFT_INVERSE);
		if (result != CUFFT_SUCCESS) { printf ("pcD INVERSE *XtExec* failed\n"); return; }



		for(int i=0; i < nGPUs ;i++)
		{
			device = p1D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));

			scal_vr_3d<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) pcD->descriptor->data[i],(cufftComplex*) p2D->descriptor->data[i],(cufftComplex*) vbD->descriptor->data[i],R,nxb,nyb,nzb/2,2.0,TRUE);


		}

		synchronize_gpus(nGPUs);



		for(int i=0; i < nGPUs ;i++)
		{
			device = p1D->descriptor->GPUs[i];
			//Set device
			checkCudaErrors(cudaSetDevice(device));
			//	STEP 12
			add_bessel_3d<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) p2D->descriptor->data[i],(cufftComplex*) p2D->descriptor->data[i],dtR,n+1,nxb,nyb,nzb/2);
			//	STEP 13
			add_bessel_3d<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) q2D->descriptor->data[i],(cufftComplex*) q2D->descriptor->data[i],dtR,n+1,nxb,nyb,nzb/2);
		}

		synchronize_gpus(nGPUs);


		for(int i=0; i < nGPUs ;i++)
		{
			device = p2D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));		//Set device
			apply_ABC_Xt<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) q2D->descriptor->data[i],att_blr,att_bfb,att_bud,nxb,nyb,nzb/2,i);
			apply_ABC_Xt<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) p2D->descriptor->data[i],att_blr,att_bfb,att_bud,nxb,nyb,nzb/2,i);
		}
		synchronize_gpus(nGPUs);


		for(int i=0; i < nGPUs ;i++)
		{
			device = p2D->descriptor->GPUs[i];
			checkCudaErrors(cudaSetDevice(device));		//Set device
			replace_2array_3d<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) p1D->descriptor->data[i],(cufftComplex*) p2D->descriptor->data[i],nxb,nyb,nzb/2);
			replace_2array_3d<<<dimGrid3D_half_before_tran,dimBlock2D>>>((cufftComplex*) q1D->descriptor->data[i],(cufftComplex*) q2D->descriptor->data[i],nxb,nyb,nzb/2);
		}
		synchronize_gpus(nGPUs);

	}

}

void rec_slice_to_data(float *slice,float *data,int ng,int it)
{
	int ig;
	for(ig=0;ig<ng;ig++)
	{
		data[it*ng+ig]=slice[ig];
	}

}


void input_file_xyz_boundary(char *filename,float *data,int nx,int ny,int nz,int bl,int bf,int bu,int nxb,int nyb,int nzb)
{
	FILE *fp;
	int ix,iy,iz;
	if((fp=fopen(filename,"rb"))==NULL)
	{	
		printf("Can not oen this file : %s\n",filename);
		exit(0);
	}

	fp=fopen(filename,"rb");
	for(ix=0;ix<nx;ix++)
	{
		for(iy=0;iy<ny;iy++)
		{
			for(iz=0;iz<nz;iz++)
			{
				fread(&data[(iz+bu)*nxb*nyb+(iy+bf)*nxb+ix+bl],sizeof(float),1,fp);
			}
		}
	}
	fclose(fp);
}


void add_pml_layers_v_h(float *v_h,int nx,int ny,int nz,int bl,int bf,int bu,int nxb,int nyb,int nzb)
{
	int ix,iy,iz;
//l	3

	for(iy=bf;iy<ny+bf;iy++)
	for(ix=0;ix<bl;ix++)
	for(iz=bu;iz<nz+bu;iz++)
	{
	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz*nyb*nxb+iy*nxb+bl];
//	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz][iy][bl];
	}

//r	4
	for(iy=bf;iy<ny+bf;iy++)
	for(ix=bl+nx;ix<nxb;ix++)
	for(iz=bu;iz<nz+bu;iz++)
	{
	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz*nyb*nxb+iy*nxb+bl+nx-1];
//	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz][iy][bl+nx-1];
	}

//f	5
	for(iy=0;iy<bf;iy++)
	for(ix=bl;ix<nx+bl;ix++)
	for(iz=bu;iz<nz+bu;iz++)
	{
	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz*nyb*nxb+bf*nxb+ix];
//	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz][bf][ix];
	}


//b	6

	for(iy=bf+ny;iy<nyb;iy++)
	for(ix=bl;ix<nx+bl;ix++)
	for(iz=bu;iz<nz+bu;iz++)
	{
	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz*nyb*nxb+(bf+ny-1)*nxb+ix];
//	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz][bf+ny-1][ix];
	}





//fl	11
	for(iy=0;iy<bf;iy++)
	for(ix=0;ix<bl;ix++)
	for(iz=bu;iz<nz+bu;iz++)
	{
	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz*nyb*nxb+bf*nxb+bl];
//	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz][bf][bl];
	}


//fr	12

	for(iy=0;iy<bf;iy++)
	for(ix=bl+nx;ix<nxb;ix++)
	for(iz=bu;iz<nz+bu;iz++)
	{
	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz*nyb*nxb+bf*nxb+bl+nx-1];
//	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz][bf][bl+nx-1];
	}




//bl	13

	for(iy=bf+ny;iy<nyb;iy++)
	for(ix=0;ix<bl;ix++)
	for(iz=bu;iz<nz+bu;iz++)
	{
	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz*nyb*nxb+(bf+ny-1)*nxb+bl];
//	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz][bf+ny-1][bl];
	}




//br	14



	for(iy=bf+ny;iy<nyb;iy++)
	for(ix=bl+nx;ix<nxb;ix++)
	for(iz=bu;iz<nz+bu;iz++)
	{
	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz*nyb*nxb+(bf+ny-1)*nxb+bl+nx-1];
//	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[iz][bf+ny-1][bl+nx-1];
	}



//u_all


	for(iy=0;iy<nyb;iy++)
	for(ix=0;ix<nxb;ix++)
	for(iz=0;iz<bu;iz++)
	{
	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[bu*nyb*nxb+iy*nxb+ix];
//	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[bu][iy][ix];
	}


//d_all


	for(iy=0;iy<nyb;iy++)
	for(ix=0;ix<nxb;ix++)
	for(iz=bu+nz;iz<nzb;iz++)
	{
	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[(bu+nz-1)*nyb*nxb+iy*nxb+ix];
//	v_h[iz*nyb*nxb+iy*nxb+ix]=v_h[bu+nz-1][iy][ix];
	}


}


