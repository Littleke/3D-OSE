// Complex data type
typedef float2 Complex;
__global__ void generate_wavelet_gpu(float *wavelet,float fm,int nt,float dt,float amp)
{
		int i;
		float tmp;
		for(i=0;i<nt;i++)
		{
			tmp=pai*fm*((float)(i-40)*dt-1.0/fm);
			tmp=tmp*tmp;
			wavelet[i]=amp*(1.0-2.0*tmp)*expf(-tmp);
		}
}

__global__ void inputdata_3d(float *data3d,int nx,int ny,int nz,float *data1d)
{

	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z*blockDim.z+threadIdx.z;
	
	int id=tz*ny*nx+ty*nx+tx;
	if(tx<nx&&ty<ny&&tz<nz)
	{

		data3d[id]=data1d[tz];
	}
}

__global__ void R2C_3D(cufftComplex *complex,float *real,int nx,int ny,int nz,bool yes)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=tz*ny*nx+ty*nx+tx;
	if(tx<nx&&ty<ny&&tz<nz)
	{
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



__global__ void Cr2R(float *real, cufftComplex *complex,int nx,int ny,int nz)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=tz*ny*nx+ty*nx+tx;
	if(tx<nx&&ty<ny&&tz<nz)
	{
			real[id]=complex[id].x;
	}
}

__global__ void Ci2R(float *real, cufftComplex *complex,int nx,int ny,int nz)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=tz*ny*nx+ty*nx+tx;
	if(tx<nx&&ty<ny&&tz<nz)
	{
			real[id]=complex[id].y;
	}
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s)
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}


static __global__ void data_devide_test(cufftComplex *data,int nx,int ny,int nz,int iGPU)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=tz*ny*nx+ty*nx+tx;

	if(tx<nx&&ty<ny&&tz<nz)
	{
		data[id].x = tx*1.0;
	}
}



static __global__ void scale_kxkz_3D(cufftComplex *data,int nx,int ny,int nz,int iGPU,float dkx,float dky,float dkz)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=tz*ny*nx+ty*nx+tx;


	float kx=dkx*((float)tx);
	float ky=dky*((float)ty);
	float kz=dkz*((float)tz);

	if(tx<nx&&ty<ny&&tz<nz)
	{
		if(tx>nx/2)
		kx=dkx*(float)(nx-tx);
		if(iGPU==1)
		ky=dky*(float)(ny-ty);
		if(tz>nz/2)
		kz=dkz*(float)(nz-tz);

		data[id].x=data[id].x*(float)sqrt(pow((double)kx,2.0)+pow((double)ky,2.0)+pow((double)kz,2.0));
		data[id].y=data[id].y*(float)sqrt(pow((double)kx,2.0)+pow((double)ky,2.0)+pow((double)kz,2.0));
	}

}


__global__ void hilbert_3d(cufftComplex *complex,int nx,int ny,int nz)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=tz*ny*nx+ty*nx+tx;

	float real,image;
	if(tz<nz/2)
	{
			real=complex[id].y*(1.0);
			image=complex[id].x*(-1.0);
			complex[id].x=real;
			complex[id].y=image;
	}
	else
	{
			real=complex[id].y*(-1.0);
			image=complex[id].x*(1.0);
			complex[id].x=real;
			complex[id].y=image;
	}
}


__global__ void hilbert_1d(cufftComplex *complex,int nt)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
	float real,image;
	if(tx<nt/2)
	{
			real=complex[tx].y*(1.0);
			image=complex[tx].x*(-1.0);
			complex[tx].x=real;
			complex[tx].y=image;
	}
	else
	{
			real=complex[tx].y*(-1.0);
			image=complex[tx].x*(1.0);
			complex[tx].x=real;
			complex[tx].y=image;
	}
}



__global__ void scale_wavelet(float *data,int nt)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;

	if(tx<nt)
	{
		data[tx]=data[tx]/(1.0*nt);
	}

}




/*
__global__ void add_wavelet_sxyz(int *sxyz_d,cufftComplex *wf1_d,int is,float *wavelet_d,int nx,int nxb,int ny,int nyb,int bl,int bf,int bu,int it)
{
	int sx,sy,sz;
	sx=sxyz_d[is]%(nx*ny)%nx;
	sy=sxyz_d[is]%(nx*ny)/nx;
	sz=sxyz_d[is]/(nx*ny);

	wf1_d[(bu+sz)*nxb*nyb+(bf+sy)*nxb+sx+bl].y+=wavelet_d[it];

}
*/
__global__ void sg_init_device(int *xyz_d, int xbeg, int ybeg, int zbeg, int jx, int jy, int jz, int number, int nx, int ny)
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



static __global__ void Xtdata_devide_value(cufftComplex *data,int nx,int ny,int nz,int iGPU, float value)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=tz*ny*nx+ty*nx+tx;

	if(tx<nx&&ty<ny&&tz<nz)
	{
		data[id].x = value;
		data[id].y = value;
	}
}
/*
__global__ void add_wavelet_sxyz(int *sxyz_d,cufftComplex *wf1_d,int is,float *wavelet_d,int nx,int nxb,int ny,int nyb,int bl,int bf,int bu,int it)
{
	int sx,sy,sz;
	sx=sxyz_d[is]%(nx*ny)%nx;
	sy=sxyz_d[is]%(nx*ny)/nx;
	sz=sxyz_d[is]/(nx*ny);

	wf1_d[(bu+sz)*nxb*nyb+(bf+sy)*nxb+sx+bl].y+=wavelet_d[it];

}
*/

__global__ void generate_att(float *data,int before,int nn,int after)
{
	int i,nnb;
	nnb=before+nn+after;
	float temp;
	for(i=0;i<before;i++)
	{
		temp=(float)(before-1-i)/(float)(before);
		data[i]=expf((-1.0)*temp*temp);
	}
	for(i=before;i<nnb-after;i++)
	{
		data[i]=1.0;
	}
	for(i=0;i<after;i++)
	{
		temp=(float)(i+1)/(float)(after);
		data[before+nn+i]=expf((-1.0)*temp*temp);
	}
}

__global__ void apply_ABC_Xt(cufftComplex *dataD,float *att_lr,float *att_fb,float *att_ud,int nx,int ny,int nz,int iGPU)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=tz*ny*nx+ty*nx+tx;

	if(tx<nx&&ty<ny&&tz<nz)
	{
		dataD[id].y=dataD[id].y*att_lr[tx];

		dataD[id].y=dataD[id].y*att_fb[ty];

		if(iGPU==0)
		{
			dataD[id].y=dataD[id].y*att_ud[tz];
		}
		else
		{
			dataD[id].y=dataD[id].y*att_ud[tz+nz];
		}

	}


}


__global__ void add_wavelet_Xt(cufftComplex *dataD,float *wavelet_d,int nxb,int nyb,int nzb_half,int bl,int bf,int bu,int sx,int sy,int sz,int it,int i)
{
	if(i==0)
	dataD[(bu+sz)*nxb*nyb+(bf+sy)*nxb+sx+bl].y+=wavelet_d[it];
	else
	dataD[(sz-nzb_half+bu)*nxb*nyb+(bf+sy)*nxb+sx+bl].y+=wavelet_d[it]; //There is a potential BUG here, when nz is odd
}


__global__ void step1to4_3d(cufftComplex *wf2,int nx,int ny,int nz,float dtR,int n)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=tz*ny*nx+ty*nx+tx;

	if(tx<nx&&ty<ny&&tz<nz)
	{
		wf2[id].x=wf2[id].y;
		wf2[id].y=jnf(n,dtR)*wf2[id].x;
	}
}



__global__ void memcpy_between_device(cufftComplex *indata,cufftComplex *outdata,int nx,int ny,int nz)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=tz*ny*nx+ty*nx+tx;

	if(tx<nx&&ty<ny&&tz<nz)
	{
		outdata[id].x=indata[id].x;
		outdata[id].y=indata[id].y;
	}
}

__global__ void memcpy_between_device_cr(cufftComplex *indata,cufftComplex *outdata,int nx,int ny,int nz)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=tz*ny*nx+ty*nx+tx;

	if(tx<nx&&ty<ny&&tz<nz)
	{
		outdata[id].x=indata[id].x;
	}
}



__global__ void scal_vr_3d(cufftComplex *input,cufftComplex *output,cufftComplex *v,float R,int nx,int ny,int nz,float scale,bool real)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=tz*ny*nx+ty*nx+tx;

	if(tx<nx&&ty<ny&&tz<nz)
	{
		if(real)
		output[id].x=output[id].x+input[id].x*scale*v[id].x/R/((float)nx*(float)ny*(float)nz*2.0);
		else
		output[id].x=input[id].x*scale*v[id].x/R/((float)nx*(float)ny*(float)nz*2.0);
	}
}


__global__ void scal_vr_3d_test(cufftComplex *input,cufftComplex *output,float v,float R,int nx,int ny,int nz,float scale,bool real)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=tz*ny*nx+ty*nx+tx;

	if(tx<nx&&ty<ny&&tz<nz)
	{
		if(real)
		output[id].x=output[id].x+input[id].x*scale*v/R/((float)nx*(float)ny*(float)nz*2.0);
		else
		output[id].x=input[id].x*scale*v/R/((float)nx*(float)ny*(float)nz*2.0);
	}
}


__global__ void add_bessel_3d(cufftComplex *input,cufftComplex *output,float dtR,int n,int nx,int ny,int nz)
{
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=tz*ny*nx+ty*nx+tx;

	if(tx<nx&&ty<ny&&tz<nz)
	{
		output[id].y=output[id].y+2.0*jnf(n,dtR)*input[id].x;
	}
}


__global__ void replace_2array_3d(cufftComplex *data1,cufftComplex *data2,int nx,int ny,int nz)
{

	float temp;
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=tz*ny*nx+ty*nx+tx;

	if(tx<nx&&ty<ny&&tz<nz)
	{
		temp=data1[id].x;
		data1[id].x=data2[id].x;
		data2[id].x=temp;
	}

}


__global__ void rec_shotdata_3d(cufftComplex *data,float *slice,int *gxyz_d,int nx,int ny,int nz,int bl,int bf,int bu,int nxb,int nyb,int nzb,int ng)
{

//	int tx = blockIdx.x*blockDim.x+threadIdx.x;

/*
	int gx,gy,gz;
	gx=gxyz_d[is]%(nx*ny)%nx;
	gy=gxyz_d[is]%(nx*ny)/nx;
	gz=gxyz_d[is]/(nx*ny);
	int id=(gz+bu)*nxb*nyb+(gy+bf)*nxb+gx+bl;
*/
	int tx = blockIdx.x*blockDim.x+threadIdx.x;
    	int ty = blockIdx.y*blockDim.y+threadIdx.y;
	int tz = blockIdx.z;
	int id=(tz+bu)*nyb*nxb+(ty+bf)*nxb+tx+bl;
	int id0=tz*ny*nx+ty*nx+tx;

	if(tx<nx&&ty<ny&&tz<1)
	{
//		slice[tx]=data[id].y;
		slice[id0]=data[id].y;
	}

}



