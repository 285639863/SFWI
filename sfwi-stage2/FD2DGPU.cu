#include "FD2DGPU.cuh"
#include <stdio.h>

#include <ctime>
#include <sys/time.h>

#ifdef VECTORIMG
	#define datatype float2
#else
	#define datatype float
#endif	

using namespace std;

FD2DGPU::FD2DGPU(const float* sou,const float dx1,const float dz1, const float dt1, const int nxpml1, const int nzpml1, \
const int allnx1,const int allnz1,const int scale1,const int pml1,const int nt1,const int nop1)
:dx(dx1),dz(dz1),dt(dt1),nxpml(nxpml1),nzpml(nzpml1),allnx(allnx1),allnz(allnz1),scale(scale1),pml(pml1),nt(nt1),nop(nop1)
{
	// if(cudaMalloc((void**)&coeff,6*sizeof(float))==cudaSuccess) printf("cuda success\n");
	// cudaMemcpy(coeff,C,6*sizeof(float),cudaMemcpyHostToDevice);

    int nDim = nxpml*nzpml;
	int nxnz = allnx*allnz;
	cudaMalloc((void**)&vel,nxnz*sizeof(float));
	cudaMalloc((void**)&vt,nDim*sizeof(float));	
	
	cudaMalloc((void**)&px,nDim*sizeof(float));
	cudaMalloc((void**)&pz,nDim*sizeof(float));		
	cudaMalloc((void**)&p,nDim*sizeof(float));
	cudaMalloc((void**)&vx,nDim*sizeof(float));
	cudaMalloc((void**)&vz,nDim*sizeof(float));            


	cudaMemset(px,0,nDim*sizeof(float));
    cudaMemset(pz,0,nDim*sizeof(float));
	cudaMemset(p,0,nDim*sizeof(float));
	cudaMemset(vx,0,nDim*sizeof(float));
	cudaMemset(vz,0,nDim*sizeof(float));

	cudaMalloc((void**)&d_k,nDim*sizeof(float));
	cudaMalloc((void**)&d_rho,nDim*sizeof(float));		
	cudaMalloc((void**)&d_damp,nDim*sizeof(float));

	cudaMemset(d_damp,0,nDim*sizeof(float));

	cudaMalloc((void**)&source,nt*sizeof(float));
	cudaMemcpy(source,sou,nt*sizeof(float),cudaMemcpyHostToDevice);


    nx = nxpml - 2*pml;
	nz = nzpml - 2*pml;

//	int ntrace = nx%scale==0 ? nx/scale+1 : nx/scale;
//	int ntrace = nx/scale;
//	cudaMalloc((void **)&receiver,ntrace*nt*sizeof(float));

    Grid_Block.grid.x=(nzpml+BLOCKDIMX-1)/BLOCKDIMX;
    Grid_Block.grid.y=(nxpml+BLOCKDIMY-1)/BLOCKDIMY;
    Grid_Block.block.x=BLOCKDIMX;
    Grid_Block.block.y=BLOCKDIMY;

    record.grid.x=(nx+BLOCKDIMX-1)/BLOCKDIMX;
    record.block.x=BLOCKDIMX;

    // velcopy.grid.x=(nz+BLOCKDIMX-1)/BLOCKDIMX;
    // velcopy.grid.y=(nx+BLOCKDIMY-1)/BLOCKDIMY;
    // velcopy.block.x=BLOCKDIMX;
    // velcopy.block.y=BLOCKDIMY;	

}


FD2DGPU::~FD2DGPU(){


	cudaFree(vel);
	cudaFree(vt);
    cudaFree(px);
    cudaFree(pz);
    cudaFree(p);
    cudaFree(vx);
    cudaFree(vz);
    cudaFree(d_k);
    cudaFree(d_rho);
    cudaFree(d_damp);
    cudaFree(source);
//  cudaFree(receiver);                            

}

__global__ void testifzero(float *vt , const int nxpml ,const int nzpml){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz<nzpml&&ix<nxpml){
		if(vt[ix*nzpml+iz]==0)printf("ix=%d,iz=%d\n",ix,iz);
	}

}


void FD2DGPU::testcopyV(float *h_v){

	cudaMemcpy(vt,h_v,sizeof(float)*nxpml*nzpml,cudaMemcpyHostToDevice);
	testifzero<<<Grid_Block.grid,Grid_Block.block>>>(vt,nxpml,nzpml);

}





__global__ void copyvelocityHtoD(float* vel,float* vt,const int pml,const int nzpml,const int ngx_min,const int allnz,const int nx, const int nz){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;	
	if(iz<nz&&ix<nx){
		vt[(ix+pml)*nzpml+iz+pml]=vel[(ngx_min+ix)*allnz+iz];
	}
	// __syncthreads();
	// if(ix<nxpml&&iz<pml){
    //     vt[ix*nzpml+iz] = vt[ix*nzpml+pml];
    //     vt[ix*nzpml+nzpml-pml+iz] = vt[ix*nzpml+nzpml-pml-1];		
	// }	
	// __syncthreads();
	// if(ix<pml&&iz<nzpml){
	// 	vt[ix*nzpml+iz] = vt[pml*nzpml+iz];
	// 	vt[(nxpml-pml+ix)*nzpml+iz] = vt[(nxpml-pml-1)*nzpml+iz];		
	// }	
}


/*
__global__ void random_boundary(float* vt,const int pml,const int nxpml,const int nzpml){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(ix<nxpml&&iz<pml){
	k = (vpmax-vpmin)/pml;

        vt[ix*nzpml+iz] = vt[ix*nzpml+pml] + (-1+2*((float)rand())/RAND_MAX)*k*iz;
        vt[ix*nzpml+nzpml-pml+iz] = vt[ix*nzpml+nzpml-pml] + (-1+2*((float)rand())/RAND_MAX)*k*iz;		
	}	
}
*/

__device__ float generate(curandState *globalState, int ind)
{
	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);// uniform distribution
//	float RANDOM = curand_normal(&localState);// normal distribution
	globalState[ind] = localState;
	return RANDOM;
}

__global__ void random_vel(curandState *globalState,float *vp,float *vs, const float vmax, const int nx,const int nz,const int nzpml,const float k,const int nxpml,const int pml,int direction_z)
{
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	int ix = threadIdx.y + blockIdx.y*blockDim.y;
	int idx = ix * blockDim.x*gridDim.x + iz;
	int dis,index,index_opposite;
	if(direction_z == 1){
		dis = nz - iz;
		index = ix*nzpml + iz;
		index_opposite = ix*nzpml+(nzpml-iz);
	}
	else{
		dis = nx - ix;
		index = ix*nzpml + iz;
		index_opposite = (nxpml-ix)*nzpml+iz;		
	}
	if (iz < nz&&ix < nx)
	{
//top left
		float vnew;
		vnew = (-1+2*generate(globalState, idx))*k*dis + vp[index];
		if(vnew>=0 && vnew<=vmax){
			vs[index] = (vs[index]/vp[index])*vnew;
			vp[index] = vnew; 		
		}
		else{
			for(;;){
				vnew = (-1+2*generate(globalState, idx))*k*dis + vp[index];
				if(vnew>=0 && vnew<=vmax){
					vs[index] = (vs[index]/vp[index])*vnew;
					vp[index] = vnew;				
					break;
				}			
			}
		}
//top

//bottom right
		vnew = (-1+2*generate(globalState, idx))*k*dis + vp[index_opposite];
		if(vnew>=0 && vnew<=vmax){
			vs[index_opposite] = (vs[index]/vp[index])*vnew;
			vp[index_opposite] = vnew;		
		}
		else{
			for(;;){
				vnew = (-1+2*generate(globalState, idx))*k*dis + vp[index_opposite];
				if(vnew>=0 && vnew<=vmax){
					vs[index_opposite] = (vs[index]/vp[index])*vnew;
					vp[index_opposite] = vnew;				
					break;
				}			
			}
		}
//bottom
	}

}



__global__ void setup_kernel(curandState *state, unsigned long seed,const int nxpml,const int pml)
{
	int iz = threadIdx.x + blockIdx.x*blockDim.x;
	int ix = threadIdx.y + blockIdx.y*blockDim.y;
	int id = ix * blockDim.x*gridDim.x + iz;
//	if(ix>nxpml||iz>pml){return;}
	curand_init(seed, id, 0, &state[id]);// initialize the state
}



__global__ void copypmlVz(float* vt,const int pml,const int nxpml,const int nzpml){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(ix<nxpml&&iz<pml){
        vt[ix*nzpml+iz] = vt[ix*nzpml+pml];
        vt[ix*nzpml+nzpml-pml+iz] = vt[ix*nzpml+nzpml-pml-1];		
	}	
}

__global__ void copypmlVx(float* vt,const int pml,const int nxpml,const int nzpml){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(ix<pml&&iz<nzpml){
		vt[ix*nzpml+iz] = vt[pml*nzpml+iz];
		vt[(nxpml-pml+ix)*nzpml+iz] = vt[(nxpml-pml-1)*nzpml+iz];		
	}	
}


__global__ void GPUcalculate_p(float *px, float *pz,float *p,float *vx,float *vz,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,const int nop, const float *k, const float* damp)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop-1||ix>nxpml-nop-1||iz<nop-1||ix<nop-1)return;
//	__syncthreads();

		float damp1 = 1 - dt*damp[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*damp[iz+ix*nzpml]/2;

		float tmp_vx = 0;
		float tmp_vz = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_vx += coeff2[i]*(vx[(ix+i)*nzpml+iz]-vx[(ix-i+1)*nzpml+iz]);
			tmp_vz += coeff2[i]*(vz[ix*nzpml+(iz+i)]-vz[ix*nzpml+(iz-i+1)]);
		}
		
		 px[iz+ix*nzpml] = (damp1*px[iz+ix*nzpml]-k[iz+ix*nzpml]*(dt/dx)*tmp_vx)/damp2;	
		 pz[iz+ix*nzpml] = (damp1*pz[iz+ix*nzpml]-k[iz+ix*nzpml]*(dt/dz)*tmp_vz)/damp2;

        p[iz+ix*nzpml] = px[iz+ix*nzpml] + pz[iz+ix*nzpml];


}

__global__ void GPUcalculate_vx(float *p,float *vx,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,const int nop,float *rho,float *damp)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop-1||ix>nxpml-nop||iz<nop||ix<nop)return;
//	__syncthreads();
		float damp1 = 1 - dt*damp[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*damp[iz+ix*nzpml]/2;
		float tmp_p = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_p += coeff2[i]*(p[(ix+i-1)*nzpml+iz]-p[(ix-i)*nzpml+iz]);

		}
		
		vx[iz+ix*nzpml] = (damp1*vx[iz+ix*nzpml]-(1.0/rho[iz+ix*nzpml])*(dt/dx)*tmp_p)/damp2;

}


__global__ void GPUcalculate_vz(float *p,float *vz,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,const int nop,float *rho,float *damp)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop-1||iz<nop||ix<nop)return;
//	__syncthreads();
		float damp1 = 1 - dt*damp[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*damp[iz+ix*nzpml]/2;
		float tmp_p = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_p += coeff2[i]*(p[ix*nzpml+(iz+i-1)]-p[ix*nzpml+(iz-i)]);

		}
		
		vz[iz+ix*nzpml] = (damp1*vz[iz+ix*nzpml]-(1.0/rho[iz+ix*nzpml])*(dt/dx)*tmp_p)/damp2;

}

__global__ void GPUapplysource(float *vz,float *sou,const int sxnum,const int nxpml,const int nzpml,const float dt,const float dx,const int it,const int pml,const float *rho)
{
//  const int index = (blockIdx.y*gridDim.x)*blockDim.x*blockDim.y+blockIdx.x*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x;
    const int index = blockIdx.x*blockDim.x+threadIdx.x;
	if(index>0)return;

//	vz[(sxnum+pml)*nzpml+pml] +=  (dt/(dx*dx*rho[(sxnum+pml)*nzpml+pml]))*sou[it];	

//	vz[(sxnum+pml)*nzpml+pml] +=  sou[it];	

//	vz[(sxnum+pml)*nzpml+pml+20] +=  sou[it];	
	
	vz[(sxnum+pml)*nzpml+pml] +=  sou[it];	
}

__global__ void GPUseisrecord_acoustic(float *p0,float *receiver,const int it,const int nt,const int nx, const int scale,const int pml,const int nzpml)
{
	const int ix = blockIdx.x*blockDim.x+threadIdx.x;
	if(ix>=nx)return;
	
	const long index = (ix+pml)*nzpml+(pml);
//	if(ix%scale==0&&ix/scale<ns_x&&iy%scale==0&&iy/scale<ns_y) receiver[(iy/scale)*xtrace*nt+(ix/scale)*nt+it] = snap[3*nDim+index];
	if(ix%scale==0) receiver[(ix/scale)*nt+it] = p0[index];			//vz		//vx

}



__global__ void GPUseisrecord(float *p0,float *p1,float *receiver,float *receiver2,const int it,const int nt,const int nx, const int scale,const int pml,const int nzpml,const int gy)
{
	const int ix = blockIdx.x*blockDim.x+threadIdx.x;
	if(ix>=nx)return;
	
	const long index = (ix+pml)*nzpml+(pml+gy);
//	if(ix%scale==0&&ix/scale<ns_x&&iy%scale==0&&iy/scale<ns_y) receiver[(iy/scale)*xtrace*nt+(ix/scale)*nt+it] = snap[3*nDim+index];
	if(ix%scale==0) receiver[(ix/scale)*nt+it] = p0[index];			//vz
	if(ix%scale==0) receiver2[(ix/scale)*nt+it] = p1[index];		//vx

}


__global__ void GPUseisrecord_mute_elastic(float *vz,float *vx,float *receiver,float *receiver2,const int it,const int nt,const int nx, const int scale,const int pml,const int nzpml,const int sxnum,const float dx,const float dz, \
	float* vp,const float dt,const int ntd,const int sy,const int gy)
{
	const int ix = blockIdx.x*blockDim.x+threadIdx.x;
	if(ix>=nx)return;
	const long index = (ix+pml)*nzpml+pml;
	float vmute = vp[(sxnum+pml)*nzpml+pml+sy];
//	if(ix%scale==0&&ix/scale<ns_x&&iy%scale==0&&iy/scale<ns_y) receiver[(iy/scale)*xtrace*nt+(ix/scale)*nt+it] = snap[3*nDim+index];
	if(ix%scale==0) {
	float a=dx*abs(ix-sxnum);
	float b=dz*abs(gy-sy);
	float t0=sqrtf(a*a+b*b)/vmute;
	int ktt=int(t0/dt)+ntd;// ntd is manually added to obtain the best muting effect.
	receiver[(ix/scale)*nt+it] = it<ktt ? 0.0 : vz[index];
	receiver2[(ix/scale)*nt+it] = it<ktt ? 0.0 : vx[index];
	}
}



__global__ void GPUseisrecord_mute(float *p,float *receiver,const int it,const int nt,const int nx, const int scale,const int pml,const int nzpml,const int sxnum,const float dx, \
	const float vmute,const float dt,const int ntd)
{
	const int ix = blockIdx.x*blockDim.x+threadIdx.x;
	if(ix>=nx)return;
	const long index = (ix+pml)*nzpml+pml;
//	if(ix%scale==0&&ix/scale<ns_x&&iy%scale==0&&iy/scale<ns_y) receiver[(iy/scale)*xtrace*nt+(ix/scale)*nt+it] = snap[3*nDim+index];
	if(ix%scale==0) {
	float a=dx*abs(ix-sxnum);
	float t0=sqrtf(a*a)/vmute;
	int ktt=int(t0/dt)+ntd;// ntd is manually added to obtain the best muting effect.
    receiver[(ix/scale)*nt+it] = it<ktt ? 0.0 : p[index];
	}
}


__global__ void GPUseisrecord_float2(float2 *p,float *receiver,const int it,const int nt,const int nx, const int scale,const int pml,const int nzpml)
{
	const int ix = blockIdx.x*blockDim.x+threadIdx.x;
	if(ix>=nx)return;
	
	const long index = (ix+pml)*nzpml+pml;
//	if(ix%scale==0&&ix/scale<ns_x&&iy%scale==0&&iy/scale<ns_y) receiver[(iy/scale)*xtrace*nt+(ix/scale)*nt+it] = snap[3*nDim+index];
	if(ix%scale==0) receiver[(ix/scale)*nt+it] = p[index].y;

}


void FD2DGPU::GPUbufferV(float *v){

		cudaMemcpy(vel,v,allnx*allnz*sizeof(float),cudaMemcpyHostToDevice);

}

void FD2DGPU::bufferVHtoD(const int ngx_min){

	copyvelocityHtoD<<<Grid_Block.grid,Grid_Block.block>>>(vel,vt,pml,nzpml,ngx_min,allnz,nx,nz);

	copypmlVz<<<Grid_Block.grid,Grid_Block.block>>>(vt,pml,nxpml,nzpml);
	copypmlVx<<<Grid_Block.grid,Grid_Block.block>>>(vt,pml,nxpml,nzpml);
}



void FD2DGPU::calculate_min(int *min_idx){

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasStatus_t stat;	
	stat = cublasIsamin(handle, nxpml*nzpml, vt, 1, min_idx); 		
	// cublas just like fortran index begin from 1 
	cublasDestroy(handle); 

}


__global__ void dampX(float *d_dampx,const int pml, const float * const vmin,float *vp,const int nxpml,const int nzpml,const float dx,float rr){

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;	
	if(ix>nxpml-1||iz>nzpml-1){ return;}
	else{
	//	__shared__ float damp[Npml];
//		float a = dx*(float)(pml-1);
//		float b = 1.5f**vmin*log(10000000.0f)/a;
//		float xa;
//		xa = (float)ix/(pml-1.0f);

//		d_damp[(pml-ix-1)*nzpml+iz] = b*xa*xa;
//		d_damp[(nxpml-pml+ix)*nzpml+iz] = b*xa*xa;

		// float rr=0.000001;
		// float rr=0.01;
		if(ix<pml){
			d_dampx[ix*nzpml+iz] = log10(1/rr)*(5.0*vp[ix*nzpml+iz]/(2.0*pml))*powf(1.0*(pml-ix)/pml,4.0);
		}
		if(ix>nxpml-pml){
			d_dampx[ix*nzpml+iz] = log10(1/rr)*(5.0*vp[ix*nzpml+iz]/(2.0*pml))*powf(1.0*(ix-(nxpml-pml))/pml,4.0);
		}
	}
}


__global__ void dampZ(float *d_dampz,const int pml,const float * const vmin,float *vp,const int nxpml,const int nzpml,const float dz,float rr){

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;	
	if(iz>nzpml-1||ix>nxpml-1){return;}
	else{
//		float a = dz*(float)(pml-1);
//		float b = 1.5f**vmin*log(10000000.0f)/a;
//		float za;
//		za = (float)iz/(pml-1.0f);
//		if(ix>pml-iz-1&&ix<nxpml-(pml-iz)){
//			d_damp[ix*nzpml+pml-iz-1] = b*za*za;
//			d_damp[ix*nzpml+nzpml-pml+iz] = b*za*za;
//		}	
		// float rr=0.000001;
//		float rr=0.01;
		if(iz<pml){
			d_dampz[ix*nzpml+iz] = log10(1/rr)*(5.0*vp[ix*nzpml+iz]/(2.0*pml))*powf(1.0*(pml-iz)/pml,4.0);
		}
		if(iz>nzpml-pml){
			d_dampz[ix*nzpml+iz] = log10(1/rr)*(5.0*vp[ix*nzpml+iz]/(2.0*pml))*powf(1.0*(iz-(nzpml-pml))/pml,4.0);
		}	
	}
}

__global__ void GPUcalculate_k_rho(float *rho,float *k,float *vt,const int nxpml,const int nzpml){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;	
	if(iz>nzpml-1||ix>nxpml-1)return;
	rho[ix*nzpml+iz] = 1000;
	k[ix*nzpml+iz]=vt[ix*nzpml+iz]*vt[ix*nzpml+iz]*rho[ix*nzpml+iz];
}


void FD2DGPU::calculate_damp_k_rho(){

	int minid;

	FD2DGPU::calculate_min(&minid);

	printf("minid=%d\n",minid);
	float rr=0.000001;

	dampX<<<Grid_Block.grid,Grid_Block.block>>>(d_damp,pml,&vt[minid-1],vt,nxpml,nzpml,dx,rr);
	dampZ<<<Grid_Block.grid,Grid_Block.block>>>(d_damp,pml,&vt[minid-1],vt,nxpml,nzpml,dz,rr);

	GPUcalculate_k_rho<<<Grid_Block.grid,Grid_Block.block>>>(d_rho,d_k,vt,nxpml,nzpml);

}



void FD2DGPU::calculateP(){

    GPUcalculate_p<<<Grid_Block.grid,Grid_Block.block>>>(px,pz,p,vx,vz,nxpml,nzpml,dt,dx,dz,nop,d_k,d_damp);
}

void FD2DGPU::calculateV(){

	
    GPUcalculate_vx<<<Grid_Block.grid,Grid_Block.block>>>(p,vx,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp);
    GPUcalculate_vz<<<Grid_Block.grid,Grid_Block.block>>>(p,vz,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp);
}

void FD2DGPU::addS(const int sxnum,const int it){

    GPUapplysource<<<1,1>>>(vz,source,sxnum,nxpml,nzpml,dt,dx,it,pml,d_rho);
}

void FD2DGPU::recordD(const int scale,const int pml,const int it){
	
    GPUseisrecord_acoustic<<<record.grid,record.block>>>(p,receiver,it,nt,nx,scale,pml,nzpml);

}


void FD2DGPU::FD_initial(const int nDim){
	cudaMemset(px,0,nDim*sizeof(float));
    cudaMemset(pz,0,nDim*sizeof(float));
	cudaMemset(p,0,nDim*sizeof(float));
	cudaMemset(vx,0,nDim*sizeof(float));
	cudaMemset(vz,0,nDim*sizeof(float));
}
/*
void FD(FD2DGPU &fdtdgpu,const int sxnum,float *rec,const int nDim){


	fdtdgpu.FD_initial(nDim);

	// cudaMemset(fdtdgpu.px,0,nDim*sizeof(float));
    // cudaMemset(fdtdgpu.pz,0,nDim*sizeof(float));
	// cudaMemset(fdtdgpu.p,0,nDim*sizeof(float));
	// cudaMemset(fdtdgpu.vx,0,nDim*sizeof(float));
	// cudaMemset(fdtdgpu.vz,0,nDim*sizeof(float));

	// cudaStream_t streams[2];
	// for (int i = 0; i < 2; i++) {
  	// cudaStreamCreate(&streams[i]);
	// }
	fdtdgpu.calculate_damp_k_rho();

    int it;
	for (it=0; it<fdtdgpu.nt; ++it)    
    {
        fdtdgpu.calculateP();
        fdtdgpu.calculateV();
        fdtdgpu.addS(sxnum,it);
        fdtdgpu.recordD(fdtdgpu.scale,fdtdgpu.pml,it);       
    }

    cudaMemcpy(rec,fdtdgpu.receiver,fdtdgpu.nx*fdtdgpu.nt*sizeof(float)/fdtdgpu.scale,cudaMemcpyDeviceToHost);

}
*/





//   ************************************* ELASTIC  ***************************************************
__global__ void GPUcalculate_elastic_vx_back(float *txx,float *txz,float *vx,float *vx_x,float *vx_z,float *tzz,float *lamda,float *miu,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,\
									const int nop,float *rho,float *dampx,float *dampz,int direction)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>=nzpml-nop||ix>=nxpml-nop||iz<nop||ix<nop)return;
//	__syncthreads();
		float damp1 = 1 - dt*dampx[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*dampx[iz+ix*nzpml]/2;
		float damp3 = 1 - dt*dampz[iz+ix*nzpml]/2;
		float damp4 = 1 + dt*dampz[iz+ix*nzpml]/2;

		float tmp_txx = 0;
		float tmp_txz = 0;
		float tmp_tzz = 0;

///////////////////////// lamda and miu is out	of partial differential 	
// #pragma unroll 4
// 		for(int i=1;i<=nop;i++)
// 		{		
// 			tmp_txx += coeff2[i]*(txx[(ix+i)*nzpml+iz]-txx[(ix-i+1)*nzpml+iz]);
// 			tmp_txz += coeff2[i]*(txz[ix*nzpml+(iz+i-1)]-txz[ix*nzpml+(iz-i)]);
// 			tmp_tzz += coeff2[i]*(tzz[(ix+i)*nzpml+iz]-tzz[(ix-i+1)*nzpml+iz]);			
// 		}
		
// 		vx_x[iz+ix*nzpml] = (damp1*vx_x[iz+ix*nzpml]+direction*(1.0/rho[iz+ix*nzpml])*(lamda[iz+ix*nzpml]+2*miu[iz+ix*nzpml])*(dt/dx)*tmp_txx \
// 							+direction*(1.0/rho[iz+ix*nzpml])*(lamda[iz+ix*nzpml])*(dt/dx)*tmp_tzz)/damp2;
// 		vx_z[iz+ix*nzpml] = (damp3*vx_z[iz+ix*nzpml]+direction*(1.0/rho[iz+ix*nzpml])*miu[iz+ix*nzpml]*(dt/dz)*tmp_txz)/damp4;					
// 		vx[iz+ix*nzpml] = vx_x[iz+ix*nzpml] + vx_z[iz+ix*nzpml];				//TODO : add source   divide  rho

///////////////////////// lamda miu is in partial differential 
#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{		
			tmp_txx += coeff2[i]*((lamda[(ix+i)*nzpml+iz]+2*miu[(ix+i)*nzpml+iz])*txx[(ix+i)*nzpml+iz]-(lamda[(ix-i+1)*nzpml+iz]+2*miu[(ix-i+1)*nzpml+iz])*txx[(ix-i+1)*nzpml+iz]);
			tmp_txz += coeff2[i]*(miu[ix*nzpml+(iz+i-1)]*txz[ix*nzpml+(iz+i-1)]-miu[ix*nzpml+(iz-i)]*txz[ix*nzpml+(iz-i)]);				
			tmp_tzz += coeff2[i]*((lamda[(ix+i)*nzpml+iz])*tzz[(ix+i)*nzpml+iz]-(lamda[(ix-i+1)*nzpml+iz])*tzz[(ix-i+1)*nzpml+iz]);			
		}
		
		vx_x[iz+ix*nzpml] = (damp1*vx_x[iz+ix*nzpml]+direction*(1.0/rho[iz+ix*nzpml])*(dt/dx)*tmp_txx \
							+direction*(1.0/rho[iz+ix*nzpml])*(dt/dx)*tmp_tzz)/damp2;
		vx_z[iz+ix*nzpml] = (damp3*vx_z[iz+ix*nzpml]+direction*(1.0/rho[iz+ix*nzpml])*(dt/dz)*tmp_txz)/damp4;					
		vx[iz+ix*nzpml] = vx_x[iz+ix*nzpml] + vx_z[iz+ix*nzpml];	
}


__global__ void GPUcalculate_elastic_vz_back(float *tzz,float *txz,float *vz,float *vz_x,float *vz_z,float *txx,float *lamda,float *miu,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,\
								const int nop,float *rho,float *dampx,float *dampz,int direction)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>=nzpml-nop||ix>=nxpml-nop||iz<nop||ix<nop)return;
//	__syncthreads();
		float damp1 = 1 - dt*dampx[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*dampx[iz+ix*nzpml]/2;
		float damp3 = 1 - dt*dampz[iz+ix*nzpml]/2;
		float damp4 = 1 + dt*dampz[iz+ix*nzpml]/2;

		float tmp_tzz = 0;
		float tmp_txz = 0;
		float tmp_txx = 0;		

///////////////////////// lamda and miu is out	of partial differential 
// #pragma unroll 4
// 		for(int i=1;i<=nop;i++)
// 		{		
// 			tmp_tzz += coeff2[i]*(tzz[ix*nzpml+(iz+i)]-tzz[ix*nzpml+(iz-i+1)]);
// 			tmp_txz += coeff2[i]*(txz[(ix+i-1)*nzpml+iz]-txz[(ix-i)*nzpml+iz]);
// 			tmp_txx += coeff2[i]*(txx[ix*nzpml+(iz+i)]-txx[ix*nzpml+(iz-i+1)]);			
// 		}
		
// 		vz_x[iz+ix*nzpml] = (damp1*vz_x[iz+ix*nzpml]+direction*(1.0/rho[iz+ix*nzpml])*miu[iz+ix*nzpml]*(dt/dx)*tmp_txz)/damp2;		
// 		vz_z[iz+ix*nzpml] = (damp3*vz_z[iz+ix*nzpml]+direction*(1.0/rho[iz+ix*nzpml])*(lamda[iz+ix*nzpml]+2*miu[iz+ix*nzpml])*(dt/dz)*tmp_tzz \
// 							+direction*(1.0/rho[iz+ix*nzpml])*(lamda[iz+ix*nzpml])*(dt/dz)*tmp_txx)/damp4;
// 		vz[iz+ix*nzpml] = vz_x[iz+ix*nzpml] + vz_z[iz+ix*nzpml];

///////////////////////// lamda miu is in partial differential   
#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{		
			tmp_tzz += coeff2[i]*((lamda[ix*nzpml+(iz+i)]+2*miu[ix*nzpml+(iz+i)])*tzz[ix*nzpml+(iz+i)]-(lamda[ix*nzpml+(iz-i+1)]+2*miu[ix*nzpml+(iz-i+1)])*tzz[ix*nzpml+(iz-i+1)]);
			tmp_txz += coeff2[i]*(miu[(ix+i-1)*nzpml+iz]*txz[(ix+i-1)*nzpml+iz]-miu[(ix-i)*nzpml+iz]*txz[(ix-i)*nzpml+iz]);				//TODO: miu and txz are not in a same grid point
			tmp_txx += coeff2[i]*((lamda[ix*nzpml+(iz+i)])*txx[ix*nzpml+(iz+i)]-(lamda[ix*nzpml+(iz-i+1)])*txx[ix*nzpml+(iz-i+1)]);			
		}
		
		vz_x[iz+ix*nzpml] = (damp1*vz_x[iz+ix*nzpml]+direction*(1.0/rho[iz+ix*nzpml])*(dt/dx)*tmp_txz)/damp2;		
		vz_z[iz+ix*nzpml] = (damp3*vz_z[iz+ix*nzpml]+direction*(1.0/rho[iz+ix*nzpml])*(dt/dz)*tmp_tzz \
							+direction*(1.0/rho[iz+ix*nzpml])*(dt/dz)*tmp_txx)/damp4;
		vz[iz+ix*nzpml] = vz_x[iz+ix*nzpml] + vz_z[iz+ix*nzpml];
}



__global__ void GPUcalculate_elastic_txx_tzz_born(float *txx, float *tzz,float *txx_x,float *txx_z,float *tzz_x,float *tzz_z,float *vx,float *vz,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,\
	const int nop,float *lamda,float *miu,float* dampx,float* dampz,int direction)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;
//	__syncthreads();

		float damp1 = 1 - dt*dampx[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*dampx[iz+ix*nzpml]/2;
		float damp3 = 1 - dt*dampz[iz+ix*nzpml]/2;
		float damp4 = 1 + dt*dampz[iz+ix*nzpml]/2;

		float tmp_vx = 0;
		float tmp_vz = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_vx += coeff2[i]*(vx[(ix+i)*nzpml+iz]-vx[(ix-i+1)*nzpml+iz]);
			tmp_vz += coeff2[i]*(vz[ix*nzpml+(iz+i-1)]-vz[ix*nzpml+(iz-i)]);
		}
		 txx_x[iz+ix*nzpml] = (damp1*txx_x[iz+ix*nzpml]+direction*(lamda[iz+ix*nzpml]+2*miu[iz+ix*nzpml])*(dt/dx)*tmp_vx)/damp2;
		 txx_z[iz+ix*nzpml] = (damp3*txx_z[iz+ix*nzpml]+direction*lamda[iz+ix*nzpml]*(dt/dz)*tmp_vz)/damp4;
		 tzz_x[iz+ix*nzpml] = (damp1*tzz_x[iz+ix*nzpml]+direction*lamda[iz+ix*nzpml]*(dt/dx)*tmp_vx)/damp2;
		 tzz_z[iz+ix*nzpml] = (damp3*tzz_z[iz+ix*nzpml]+direction*(lamda[iz+ix*nzpml]+2*miu[iz+ix*nzpml])*(dt/dz)*tmp_vz)/damp4;
		 txx[iz+ix*nzpml] = txx_x[iz+ix*nzpml] + txx_z[iz+ix*nzpml];
 		 tzz[iz+ix*nzpml] = tzz_x[iz+ix*nzpml] + tzz_z[iz+ix*nzpml];

//		 txx[iz+ix*nzpml] = txx[iz+ix*nzpml]+(lamda[iz+ix*nzpml]+2*miu[iz+ix*nzpml])*(dt/dx)*tmp_vx+lamda[iz+ix*nzpml]*(dt/dz)*tmp_vz;
//		 tzz[iz+ix*nzpml] = tzz[iz+ix*nzpml]+lamda[iz+ix*nzpml]*(dt/dx)*tmp_vx+(lamda[iz+ix*nzpml]+2*miu[iz+ix*nzpml])*(dt/dz)*tmp_vz;

// lamda+2miu=c11    lamda=c12
}




__global__ void GPUcalculate_elastic_txx_tzz(float *txx, float *tzz,float *txx_x,float *txx_z,float *tzz_x,float *tzz_z,float *vx,float *vz,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,\
	const int nop,float *lamda,float *miu,float* dampx,float* dampz,int direction)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>=nzpml-nop||ix>=nxpml-nop||iz<nop||ix<nop)return;
//	__syncthreads();

		float damp1 = 1 - dt*dampx[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*dampx[iz+ix*nzpml]/2;
		float damp3 = 1 - dt*dampz[iz+ix*nzpml]/2;
		float damp4 = 1 + dt*dampz[iz+ix*nzpml]/2;

		float tmp_vx = 0;
		float tmp_vz = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			// tmp_vx += coeff2[i]*(vx[(ix+i)*nzpml+iz]-vx[(ix-i+1)*nzpml+iz]);
			// tmp_vz += coeff2[i]*(vz[ix*nzpml+(iz+i-1)]-vz[ix*nzpml+(iz-i)]);
			tmp_vx += coeff2[i]*(vx[(ix+i-1)*nzpml+iz]-vx[(ix-i)*nzpml+iz]);
			tmp_vz += coeff2[i]*(vz[ix*nzpml+(iz+i-1)]-vz[ix*nzpml+(iz-i)]);
		}
		
		 txx_x[iz+ix*nzpml] = (damp1*txx_x[iz+ix*nzpml]+direction*(dt/dx)*tmp_vx)/damp2;

		 tzz_z[iz+ix*nzpml] = (damp3*tzz_z[iz+ix*nzpml]+direction*(dt/dz)*tmp_vz)/damp4;
		 txx[iz+ix*nzpml] = txx_x[iz+ix*nzpml];
 		 tzz[iz+ix*nzpml] = tzz_z[iz+ix*nzpml];

//		 txx[iz+ix*nzpml] = txx[iz+ix*nzpml]+(lamda[iz+ix*nzpml]+2*miu[iz+ix*nzpml])*(dt/dx)*tmp_vx+lamda[iz+ix*nzpml]*(dt/dz)*tmp_vz;
//		 tzz[iz+ix*nzpml] = tzz[iz+ix*nzpml]+lamda[iz+ix*nzpml]*(dt/dx)*tmp_vx+(lamda[iz+ix*nzpml]+2*miu[iz+ix*nzpml])*(dt/dz)*tmp_vz;

// lamda+2miu=c11    lamda=c12
}

__global__ void born_add_source(float *b_txx,float *b_tzz,float *b_txz,float *b_txx_x,float *b_txx_z,float *b_tzz_x,float *b_tzz_z,float *b_txz_x,float *b_txz_z,\
		float *vx,float *vz,float *delta_mp,float *delta_ms,float *lamda,float *miu,float *vp,float *vs,\
		const int nxpml,const int nzpml,const int nop,const float dx,const float dz){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	const int index = ix*nzpml+iz;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;

		float tmp_vx_x = 0;
		float tmp_vz_z = 0;
		float tmp_vx_z = 0;
		float tmp_vz_x = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_vx_x += coeff2[i]*(vx[(ix+i)*nzpml+iz]-vx[(ix-i+1)*nzpml+iz]);
			tmp_vz_z += coeff2[i]*(vz[ix*nzpml+(iz+i-1)]-vz[ix*nzpml+(iz-i)]);			
			tmp_vx_z += coeff2[i]*(vx[ix*nzpml+(iz+i)]-vx[ix*nzpml+(iz-i+1)]);
			tmp_vz_x += coeff2[i]*(vz[(ix+i-1)*nzpml+iz]-vz[(ix-i)*nzpml+iz]);
		}

			// tmp_vx_x = (vx[(ix+1)*nzpml+iz]-vx[ix*nzpml+iz]);
			// tmp_vz_z = (vz[ix*nzpml+iz+1]-vz[ix*nzpml+iz]);			
			// tmp_vx_z = (vx[ix*nzpml+iz+1]-vx[ix*nzpml+iz]);
			// tmp_vz_x = (vz[(ix+1)*nzpml+iz]-vz[ix*nzpml+iz]);

		b_txx_x[index] += 2*(lamda[index]+2*miu[index])*(delta_mp[index]/vp[index])*tmp_vx_x/dx;
		 
		b_txx_z[index] += 2*((lamda[index]+2*miu[index])*(delta_mp[index]/vp[index])-2*miu[index]*delta_ms[index]/vs[index])*tmp_vz_z/dz;

		b_tzz_x[index] += 2*((lamda[index]+2*miu[index])*(delta_mp[index]/vp[index])-2*miu[index]*delta_ms[index]/vs[index])*tmp_vx_x/dx;

		b_tzz_z[index] += 2*(lamda[index]+2*miu[index])*(delta_mp[index]/vp[index])*tmp_vz_z/dz;

		b_txz_x[index] += 2*miu[index]*(delta_ms[index]/vs[index])*tmp_vz_x/dx;
		
		b_txz_z[index] += 2*miu[index]*(delta_ms[index]/vs[index])*tmp_vx_z/dz;	


		// b_txx[index] += 2*(lamda[index]+2*miu[index])*(delta_mp[index]/vp[index])*tmp_vx_x/dx \
		// 	+ 2*((lamda[index]+2*miu[index])*(delta_mp[index]/vp[index])-2*miu[index]*delta_ms[index]/vs[index])*tmp_vz_z/dz;

		// b_tzz[index] += 2*((lamda[index]+2*miu[index])*(delta_mp[index]/vp[index])-2*miu[index]*delta_ms[index]/vs[index])*tmp_vx_x/dx \
		// 	+ 2*(lamda[index]+2*miu[index])*(delta_mp[index]/vp[index])*tmp_vz_z/dz;

		// b_txz[index] += 2*miu[index]*(delta_ms[index]/vs[index])*tmp_vz_x/dx + 2*miu[index]*(delta_ms[index]/vs[index])*tmp_vx_z/dz;	

		// b_txx[index] += 2*(lamda[index]+2*miu[index])*(delta_mp[index])*tmp_vx_x/dx \
		// 	+ 2*((lamda[index]+2*miu[index])*(delta_mp[index])-2*miu[index]*delta_ms[index])*tmp_vz_z/dz;

		// b_tzz[index] += 2*((lamda[index]+2*miu[index])*(delta_mp[index])-2*miu[index]*delta_ms[index])*tmp_vx_x/dx \
		// 	+ 2*(lamda[index]+2*miu[index])*(delta_mp[index])*tmp_vz_z/dz;

		// b_txz[index] += 2*miu[index]*(delta_ms[index])*tmp_vz_x/dx + 2*miu[index]*(delta_ms[index])*tmp_vx_z/dz;	
}



__global__ void GPUcalculate_elastic_txz_born(float *txz,float *txz_x,float *txz_z,float *vx,float *vz,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,\
	const int nop,float *miu,const float* dampx,float *dampz,int direction)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;
//	__syncthreads();

		float damp1 = 1 - dt*dampx[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*dampx[iz+ix*nzpml]/2;
		float damp3 = 1 - dt*dampz[iz+ix*nzpml]/2;
		float damp4 = 1 + dt*dampz[iz+ix*nzpml]/2;

		float tmp_vx = 0;
		float tmp_vz = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_vx += coeff2[i]*(vx[ix*nzpml+(iz+i)]-vx[ix*nzpml+(iz-i+1)]);
			tmp_vz += coeff2[i]*(vz[(ix+i-1)*nzpml+iz]-vz[(ix-i)*nzpml+iz]);
		}
		
		  txz_x[iz+ix*nzpml] = (damp1*txz_x[iz+ix*nzpml]+direction*(dt/dx)*miu[iz+ix*nzpml]*tmp_vz)/damp2;
		  txz_z[iz+ix*nzpml] = (damp3*txz_z[iz+ix*nzpml]+direction*(dt/dz)*miu[iz+ix*nzpml]*tmp_vx)/damp4;
		  txz[iz+ix*nzpml] = txz_x[iz+ix*nzpml] + txz_z[iz+ix*nzpml];

//		  txz[iz+ix*nzpml] = txz[iz+ix*nzpml]+miu[iz+ix*nzpml]*(dt/dx)*tmp_vz+miu[iz+ix*nzpml]*(dt/dz)*tmp_vx;			 
// lamda+2miu=c11    lamda=c12     

}






__global__ void GPUcalculate_elastic_txz(float *txz,float *txz_x,float *txz_z,float *vx,float *vz,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,\
	const int nop,float *miu,const float* dampx,float *dampz,int direction)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
//	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;
	if(iz>=nzpml-nop||ix>=nxpml-nop||iz<nop||ix<nop)return;
//	__syncthreads();

		float damp1 = 1 - dt*dampx[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*dampx[iz+ix*nzpml]/2;
		float damp3 = 1 - dt*dampz[iz+ix*nzpml]/2;
		float damp4 = 1 + dt*dampz[iz+ix*nzpml]/2;

		float tmp_vx = 0;
		float tmp_vz = 0;

#pragma unroll
		for(int i=1;i<=nop;i++)
		{
			// tmp_vx += coeff2[i]*(vx[ix*nzpml+(iz+i)]-vx[ix*nzpml+(iz-i+1)]);
			// tmp_vz += coeff2[i]*(vz[(ix+i-1)*nzpml+iz]-vz[(ix-i)*nzpml+iz]);
			tmp_vx += coeff2[i]*(vx[ix*nzpml+(iz+i)]-vx[ix*nzpml+(iz-i+1)]);
			tmp_vz += coeff2[i]*(vz[(ix+i)*nzpml+iz]-vz[(ix-i+1)*nzpml+iz]);
		}
		
		  txz_x[iz+ix*nzpml] = (damp1*txz_x[iz+ix*nzpml]+direction*(dt/dx)*tmp_vz)/damp2;
		  txz_z[iz+ix*nzpml] = (damp3*txz_z[iz+ix*nzpml]+direction*(dt/dz)*tmp_vx)/damp4;
		  txz[iz+ix*nzpml] = txz_x[iz+ix*nzpml] + txz_z[iz+ix*nzpml];


//		  txz[iz+ix*nzpml] = txz[iz+ix*nzpml]+miu[iz+ix*nzpml]*(dt/dx)*tmp_vz+miu[iz+ix*nzpml]*(dt/dz)*tmp_vx;			 
// lamda+2miu=c11    lamda=c12     

}





//

__global__ void GPUcalculate_elastic_vx(float *txx,float *txz,float *vx,float *vx_x,float *vx_z,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,\
									const int nop,float *rho,float *dampx,float *dampz,int direction,int pml)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	// if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;
	if(iz>=nzpml-pml||ix>=nxpml-pml||iz<pml||ix<pml)return;
//	__syncthreads();
		float damp1 = 1 - dt*dampx[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*dampx[iz+ix*nzpml]/2;
		float damp3 = 1 - dt*dampz[iz+ix*nzpml]/2;
		float damp4 = 1 + dt*dampz[iz+ix*nzpml]/2;

		float tmp_txx = 0;
		float tmp_txz = 0;
#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			// tmp_txx += coeff2[i]*(txx[(ix+i-1)*nzpml+iz]-txx[(ix-i)*nzpml+iz]);
			// tmp_txz += coeff2[i]*(txz[ix*nzpml+(iz+i-1)]-txz[ix*nzpml+(iz-i)]);
			tmp_txx += coeff2[i]*(txx[(ix+i)*nzpml+iz]-txx[(ix-i+1)*nzpml+iz]);
			tmp_txz += coeff2[i]*(txz[ix*nzpml+(iz+i-1)]-txz[ix*nzpml+(iz-i)]);
		}
		
		vx_x[iz+ix*nzpml] = (damp1*vx_x[iz+ix*nzpml]+direction*(1.0/rho[iz+ix*nzpml])*(dt/dx)*tmp_txx)/damp2;
		vx_z[iz+ix*nzpml] = (damp3*vx_z[iz+ix*nzpml]+direction*(1.0/rho[iz+ix*nzpml])*(dt/dz)*tmp_txz)/damp4;
		// vx_x[iz+ix*nzpml] = vx_x[iz+ix*nzpml]+(1.0/rho[iz+ix*nzpml])*(dt/dx)*tmp_txx;
		// vx_z[iz+ix*nzpml] = vx_z[iz+ix*nzpml]+(1.0/rho[iz+ix*nzpml])*(dt/dz)*tmp_txz;
		vx[iz+ix*nzpml] = vx_x[iz+ix*nzpml] + vx_z[iz+ix*nzpml];

// 		float tmp_vx_x = 0;
// 		float tmp_vx_z = 0;
// #pragma unroll 4
// 		for(int i=1;i<=nop;i++)
// 		{
// 			tmp_vx_x += coeff2[i]*(vx[(ix+i-1)*nzpml+iz]-vx[(ix-i)*nzpml+iz]);
// 			tmp_vx_z += coeff2[i]*(vx[ix*nzpml+(iz+i-1)]-vx[ix*nzpml+(iz-i)]);
// 		}
// 		vx_gx[iz+ix*nzpml] = tmp_vx_x/dx;
// 		vx_gz[iz+ix*nzpml] = tmp_vx_z/dz;

}


__global__ void GPUcalculate_elastic_vz(float *tzz,float *txz,float *vz,float *vz_x,float *vz_z,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,\
								const int nop,float *rho,float *dampx,float *dampz,int direction,int pml)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	// if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;
	if(iz>=nzpml-pml||ix>=nxpml-pml||iz<pml||ix<pml)return;
//	__syncthreads();
		float damp1 = 1 - dt*dampx[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*dampx[iz+ix*nzpml]/2;
		float damp3 = 1 - dt*dampz[iz+ix*nzpml]/2;
		float damp4 = 1 + dt*dampz[iz+ix*nzpml]/2;

		float tmp_tzz = 0;
		float tmp_txz = 0;
#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			// tmp_tzz += coeff2[i]*(tzz[ix*nzpml+(iz+i)]-tzz[ix*nzpml+(iz-i+1)]);
			// tmp_txz += coeff2[i]*(txz[(ix+i)*nzpml+iz]-txz[(ix-i+1)*nzpml+iz]);
			tmp_tzz += coeff2[i]*(tzz[ix*nzpml+(iz+i)]-tzz[ix*nzpml+(iz-i+1)]);
			tmp_txz += coeff2[i]*(txz[(ix+i-1)*nzpml+iz]-txz[(ix-i)*nzpml+iz]);
		}
		
		vz_x[iz+ix*nzpml] = (damp1*vz_x[iz+ix*nzpml]+direction*(1.0/rho[iz+ix*nzpml])*(dt/dx)*tmp_txz)/damp2;		
		vz_z[iz+ix*nzpml] = (damp3*vz_z[iz+ix*nzpml]+direction*(1.0/rho[iz+ix*nzpml])*(dt/dz)*tmp_tzz)/damp4;
		// vz_x[iz+ix*nzpml] = vz_x[iz+ix*nzpml]+(1.0/rho[iz+ix*nzpml])*(dt/dz)*tmp_tzz;
		// vz_z[iz+ix*nzpml] = vz_z[iz+ix*nzpml]+(1.0/rho[iz+ix*nzpml])*(dt/dx)*tmp_txz;
		vz[iz+ix*nzpml] = vz_x[iz+ix*nzpml] + vz_z[iz+ix*nzpml];

// 		float tmp_vz_x = 0;
// 		float tmp_vz_z = 0;
// #pragma unroll 4
// 		for(int i=1;i<=nop;i++)
// 		{
// 			tmp_vz_x += coeff2[i]*(vz[ix*nzpml+(iz+i)]-vz[ix*nzpml+(iz-i+1)]);
// 			tmp_vz_z += coeff2[i]*(vz[(ix+i)*nzpml+iz]-vz[(ix-i+1)*nzpml+iz]);
// 		}
// 		vz_gx[iz+ix*nzpml] = tmp_vz_x/dx;
// 		vz_gz[iz+ix*nzpml] = tmp_vz_z/dz;

}


__global__ void GPUcalculate_elastic_txx_tzz_grad(float *txx, float *tzz,float *txx_x,float *txx_z,float *tzz_x,float *tzz_z,float *vx,float *vz,float *vx_gx,float *vz_gz,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,\
	const int nop,float *lamda,float *miu,float* dampx,float* dampz,int direction,int pml)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	// if(iz>=nzpml-nop||ix>=nxpml-nop||iz<nop||ix<nop)return;
	if(iz>=nzpml-pml||ix>=nxpml-pml||iz<pml||ix<pml)return;	
//	__syncthreads();

		float damp1 = 1 - dt*dampx[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*dampx[iz+ix*nzpml]/2;
		float damp3 = 1 - dt*dampz[iz+ix*nzpml]/2;
		float damp4 = 1 + dt*dampz[iz+ix*nzpml]/2;

		float tmp_vx = 0;
		float tmp_vz = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_vx += coeff2[i]*(vx[(ix+i-1)*nzpml+iz]-vx[(ix-i)*nzpml+iz]);
			tmp_vz += coeff2[i]*(vz[ix*nzpml+(iz+i-1)]-vz[ix*nzpml+(iz-i)]);
			// tmp_vx += coeff2[i]*(vx[(ix+i)*nzpml+iz]-vx[(ix-i+1)*nzpml+iz]);
			// tmp_vz += coeff2[i]*(vz[ix*nzpml+(iz+i)]-vz[ix*nzpml+(iz-i+1)]);			
		}
		
		 txx_x[iz+ix*nzpml] = (damp1*txx_x[iz+ix*nzpml]+direction*(lamda[iz+ix*nzpml]+2*miu[iz+ix*nzpml])*(dt/dx)*tmp_vx)/damp2;
		 txx_z[iz+ix*nzpml] = (damp3*txx_z[iz+ix*nzpml]+direction*lamda[iz+ix*nzpml]*(dt/dz)*tmp_vz)/damp4;
		 tzz_x[iz+ix*nzpml] = (damp1*tzz_x[iz+ix*nzpml]+direction*lamda[iz+ix*nzpml]*(dt/dx)*tmp_vx)/damp2;
		 tzz_z[iz+ix*nzpml] = (damp3*tzz_z[iz+ix*nzpml]+direction*(lamda[iz+ix*nzpml]+2*miu[iz+ix*nzpml])*(dt/dz)*tmp_vz)/damp4;
		 txx[iz+ix*nzpml] = txx_x[iz+ix*nzpml] + txx_z[iz+ix*nzpml];
 		 tzz[iz+ix*nzpml] = tzz_x[iz+ix*nzpml] + tzz_z[iz+ix*nzpml];


		// vx_gx[iz+ix*nzpml] = (vx[(ix+1)*nzpml+iz]-vx[ix*nzpml+iz])/dx;
		// vz_gz[iz+ix*nzpml] = (vz[ix*nzpml+(iz+1)]-vz[ix*nzpml+iz])/dz;

		vx_gx[iz+ix*nzpml] = tmp_vx/dx;
		vz_gz[iz+ix*nzpml] = tmp_vz/dz;

//		 txx[iz+ix*nzpml] = txx[iz+ix*nzpml]+(lamda[iz+ix*nzpml]+2*miu[iz+ix*nzpml])*(dt/dx)*tmp_vx+lamda[iz+ix*nzpml]*(dt/dz)*tmp_vz;
//		 tzz[iz+ix*nzpml] = tzz[iz+ix*nzpml]+lamda[iz+ix*nzpml]*(dt/dx)*tmp_vx+(lamda[iz+ix*nzpml]+2*miu[iz+ix*nzpml])*(dt/dz)*tmp_vz;

// lamda+2miu=c11    lamda=c12
}

__global__ void GPUcalculate_elastic_txz_grad(float *txz,float *txz_x,float *txz_z,float *vx,float *vz,float *vx_gz,float *vz_gx,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,\
	const int nop,float *miu,const float* dampx,float *dampz,int direction,int pml)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	// if(iz>=nzpml-nop||ix>=nxpml-nop||iz<nop||ix<nop)return;
	if(iz>=nzpml-pml||ix>=nxpml-pml||iz<pml||ix<pml)return;
//	__syncthreads();

		float damp1 = 1 - dt*dampx[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*dampx[iz+ix*nzpml]/2;
		float damp3 = 1 - dt*dampz[iz+ix*nzpml]/2;
		float damp4 = 1 + dt*dampz[iz+ix*nzpml]/2;

		float tmp_vx = 0;
		float tmp_vz = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			// tmp_vx += coeff2[i]*(vx[ix*nzpml+(iz+i)]-vx[ix*nzpml+(iz-i+1)]);
			// tmp_vz += coeff2[i]*(vz[(ix+i-1)*nzpml+iz]-vz[(ix-i)*nzpml+iz]);
			tmp_vx += coeff2[i]*(vx[ix*nzpml+(iz+i)]-vx[ix*nzpml+(iz-i+1)]);
			tmp_vz += coeff2[i]*(vz[(ix+i)*nzpml+iz]-vz[(ix-i+1)*nzpml+iz]);			
		}
		
		  txz_x[iz+ix*nzpml] = (damp1*txz_x[iz+ix*nzpml]+direction*miu[iz+ix*nzpml]*(dt/dx)*tmp_vz)/damp2;
		  txz_z[iz+ix*nzpml] = (damp3*txz_z[iz+ix*nzpml]+direction*miu[iz+ix*nzpml]*(dt/dz)*tmp_vx)/damp4;
		  txz[iz+ix*nzpml] = txz_x[iz+ix*nzpml] + txz_z[iz+ix*nzpml];

		// vx_gz[iz+ix*nzpml] = (vx[ix*nzpml+iz+1]-vx[ix*nzpml+iz])/dz;
		// vz_gx[iz+ix*nzpml] = (vz[(ix+1)*nzpml+iz]-vz[ix*nzpml+iz])/dx;

		vx_gz[iz+ix*nzpml] = tmp_vx/dz;
		vz_gx[iz+ix*nzpml] = tmp_vz/dx;



//		  txz[iz+ix*nzpml] = txz[iz+ix*nzpml]+miu[iz+ix*nzpml]*(dt/dx)*tmp_vz+miu[iz+ix*nzpml]*(dt/dz)*tmp_vx;			 
// lamda+2miu=c11    lamda=c12     

}

__global__ void elastic_dampX(float *d_damp,const int pml,const int nxpml,const int nzpml){

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;	
	if(ix>pml-1||iz>nzpml-1){ return;}
	else{
	//	__shared__ float damp[Npml];
		float a0 = 1.7f;
		float pi = 3.14159265358979f;
		float f0 = 20.0f;
		d_damp[ix*nzpml+iz] = 2*pi*a0*f0*powf((pml-ix),2)/(powf(pml,2));
		d_damp[(nxpml-ix)*nzpml+iz] = 2*pi*a0*f0*powf((pml-ix),2)/(powf(pml,2));

	}
}


__global__ void elastic_dampZ(float *d_damp,const int pml,const int nxpml,const int nzpml){

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;	
	if(iz>pml-1||ix>nxpml-1){return;}
	else{
		float a0 = 1.7f;
		float pi = 3.14159265358979f;
		float f0 = 20.0f;
		d_damp[ix*nzpml+iz] = 2*pi*a0*f0*powf((pml-iz),2)/(powf(pml,2));
		d_damp[ix*nzpml+(nzpml-iz)] = 2*pi*a0*f0*powf((pml-iz),2)/(powf(pml,2));
		}
	}



__global__ void GPUcalculate_elastic_rho_C(float *rho,float *vp,float *vs,float *lamda,float *miu,const int nxpml,const int nzpml){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;	
	if(iz>nzpml-1||ix>nxpml-1)return;
	rho[ix*nzpml+iz] = 2000;
	lamda[ix*nzpml+iz]=rho[ix*nzpml+iz]*(vp[ix*nzpml+iz]*vp[ix*nzpml+iz]-2*vs[ix*nzpml+iz]*vs[ix*nzpml+iz]);
	miu[ix*nzpml+iz]=rho[ix*nzpml+iz]*vs[ix*nzpml+iz]*vs[ix*nzpml+iz];
	
}

void FD2DGPU_ELASTIC::calculate_max(int *max_idx,float *v){

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasStatus_t stat;	
//	std::cout<<"before cublas"<<std::endl;
	stat = cublasIsamax(handle, nxpml*nzpml, v, 1, max_idx); 	
//	printf("in cublas max, nxpml=%d\t,nzpml=%d\n",nxpml,nzpml);	
	// cublas just like fortran index begin from 1 
	cublasDestroy(handle); 

}


void FD2DGPU_ELASTIC::calculate_vp_min(int *min_idx){

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasStatus_t stat;
//	printf("in cublas min. nxpml=%d\t,nzpml=%d\n",nxpml,nzpml);	
	stat = cublasIsamin(handle, nxpml*nzpml, vp, 1, min_idx); 		
//	printf("after cublas min.nxpml=%d\t,nzpml=%d\n",nxpml,nzpml);	
	// cublas just like fortran index begin from 1 
	cublasDestroy(handle); 

}

void FD2DGPU_ELASTIC::calculate_vs_min(int *min_idx){

	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasStatus_t stat;
//	printf("in cublas nxpml=%d\t,nzpml=%d\n",nxpml,nzpml);	
	stat = cublasIsamin(handle, nxpml*nzpml, vs, 1, min_idx); 		
	// cublas just like fortran index begin from 1 
	cublasDestroy(handle); 

}

void FD2DGPU_ELASTIC::LSRTM_initial(const int nDim){

	cudaMemset(s_image_pp_m,0,nDim*sizeof(float));
    cudaMemset(s_image_ps_m,0,nDim*sizeof(float));
	cudaMemset(b_theta,0,nDim*sizeof(float));
	cudaMemset(b_omega,0,nDim*sizeof(float));
	cudaMemset(b_theta_x,0,nDim*sizeof(float));
	cudaMemset(b_theta_z,0,nDim*sizeof(float));
	cudaMemset(b_omega_x,0,nDim*sizeof(float));
	cudaMemset(b_omega_z,0,nDim*sizeof(float));
	cudaMemset(b_vpx,0,nDim*sizeof(float));	
	cudaMemset(b_vpz,0,nDim*sizeof(float));
	cudaMemset(b_vsx,0,nDim*sizeof(float));
	cudaMemset(b_vsz,0,nDim*sizeof(float));
	cudaMemset(b_vx,0,nDim*sizeof(float));
	cudaMemset(b_vz,0,nDim*sizeof(float));
	cudaMemset(vpx_g,0,nDim*sizeof(float));	
	cudaMemset(vpz_g,0,nDim*sizeof(float));
	cudaMemset(vsx_g,0,nDim*sizeof(float));
	cudaMemset(vsz_g,0,nDim*sizeof(float));	
	cudaMemset(GPUrecord_scatter_z,0,nt*ntr*sizeof(float));
	cudaMemset(GPUrecord_scatter_x,0,nt*ntr*sizeof(float));	
	// cudaMemset(numerator_pp,0,nx*nz*sizeof(float));
	// cudaMemset(numerator_ps,0,nx*nz*sizeof(float));
	// cudaMemset(denominator_pp,0,nx*nz*sizeof(float));	


//gradient pp ps
	// cudaMemset(single_pp_grad,0,nx*nz*sizeof(float));
	// cudaMemset(single_ps_grad,0,nx*nz*sizeof(float));
//
	cudaMemset(b_tzz,0,nDim*sizeof(float));	
	cudaMemset(b_tzz_x,0,nDim*sizeof(float));
	cudaMemset(b_tzz_z,0,nDim*sizeof(float));
//

	cudaMemset(b_vx_bk,0,nDim*sizeof(float));	
	cudaMemset(b_vx_x_bk,0,nDim*sizeof(float));	
	cudaMemset(b_vx_z_bk,0,nDim*sizeof(float));	
	cudaMemset(b_vz_bk,0,nDim*sizeof(float));	
	cudaMemset(b_vz_x_bk,0,nDim*sizeof(float));	
	cudaMemset(b_vz_z_bk,0,nDim*sizeof(float));	
	cudaMemset(b_txx_bk,0,nDim*sizeof(float));	
	cudaMemset(b_txx_x_bk,0,nDim*sizeof(float));	
	cudaMemset(b_txx_z_bk,0,nDim*sizeof(float));	
	cudaMemset(b_tzz_bk,0,nDim*sizeof(float));	
	cudaMemset(b_tzz_x_bk,0,nDim*sizeof(float));	
	cudaMemset(b_tzz_z_bk,0,nDim*sizeof(float));	
	cudaMemset(b_txz_bk,0,nDim*sizeof(float));	
	cudaMemset(b_txz_x_bk,0,nDim*sizeof(float));	
	cudaMemset(b_txz_z_bk,0,nDim*sizeof(float));							
//

	cudaMemset(b2_vx,0,nDim*sizeof(float));	
	cudaMemset(b2_vx_x,0,nDim*sizeof(float));	
	cudaMemset(b2_vx_z,0,nDim*sizeof(float));	
	cudaMemset(b2_vz,0,nDim*sizeof(float));	
	cudaMemset(b2_vz_x,0,nDim*sizeof(float));	
	cudaMemset(b2_vz_z,0,nDim*sizeof(float));	
	cudaMemset(b2_txx,0,nDim*sizeof(float));	
	cudaMemset(b2_txx_x,0,nDim*sizeof(float));	
	cudaMemset(b2_txx_z,0,nDim*sizeof(float));	
	cudaMemset(b2_tzz,0,nDim*sizeof(float));	
	cudaMemset(b2_tzz_x,0,nDim*sizeof(float));	
	cudaMemset(b2_tzz_z,0,nDim*sizeof(float));	
	cudaMemset(b2_txz,0,nDim*sizeof(float));	
	cudaMemset(b2_txz_x,0,nDim*sizeof(float));	
	cudaMemset(b2_txz_z,0,nDim*sizeof(float));

	cudaMemset(GPUrecord_scatter2_z,0,nt*ntr*sizeof(float));
	cudaMemset(GPUrecord_scatter2_x,0,nt*ntr*sizeof(float));			
	// cudaMemset(b_numerator_pp,0,nx*nz*sizeof(float));
	// cudaMemset(b_numerator_ps,0,nx*nz*sizeof(float));
	// cudaMemset(b_denominator_pp,0,nx*nz*sizeof(float));	

}




void FD2DGPU_ELASTIC::FD_ELASTIC_initial(const int nDim){

	cudaMemset(txx,0,nDim*sizeof(float));
    cudaMemset(tzz,0,nDim*sizeof(float));
	cudaMemset(txz,0,nDim*sizeof(float));
	cudaMemset(vx,0,nDim*sizeof(float));
	cudaMemset(vz,0,nDim*sizeof(float));
	cudaMemset(vx_x,0,nDim*sizeof(float));
	cudaMemset(vx_z,0,nDim*sizeof(float));
	cudaMemset(vz_x,0,nDim*sizeof(float));
	cudaMemset(vz_z,0,nDim*sizeof(float));	
	cudaMemset(txx_x,0,nDim*sizeof(float));
	cudaMemset(txx_z,0,nDim*sizeof(float));
	cudaMemset(tzz_x,0,nDim*sizeof(float));
	cudaMemset(tzz_z,0,nDim*sizeof(float));
	cudaMemset(txz_x,0,nDim*sizeof(float));
	cudaMemset(txz_z,0,nDim*sizeof(float));	

}


void FD2DGPU_ELASTIC::BK_ELASTIC_initial(const int nDim){

	cudaMemset(theta_bk,0,nDim*sizeof(float));
    cudaMemset(omega_bk,0,nDim*sizeof(float));
	cudaMemset(theta_x_bk,0,nDim*sizeof(float));
	cudaMemset(theta_z_bk,0,nDim*sizeof(float));
	cudaMemset(omega_x_bk,0,nDim*sizeof(float));
	cudaMemset(omega_z_bk,0,nDim*sizeof(float));
	cudaMemset(vpx_bk,0,nDim*sizeof(float));
	cudaMemset(vpz_bk,0,nDim*sizeof(float));
	cudaMemset(vsx_bk,0,nDim*sizeof(float));	
	cudaMemset(vsz_bk,0,nDim*sizeof(float));
	cudaMemset(vx_bk,0,nDim*sizeof(float));
	cudaMemset(vz_bk,0,nDim*sizeof(float));
//
	cudaMemset(tzz_bk,0,nDim*sizeof(float));
	cudaMemset(vp_bk,0,nDim*sizeof(float));				//tzz_x_bk
	cudaMemset(vs_bk,0,nDim*sizeof(float));				//tzz_z_bk
//
	cudaMemset(single_image,0,nx*nz*sizeof(float));
	cudaMemset(single_image_ps,0,nx*nz*sizeof(float));

}


void FD2DGPU_ELASTIC::calculate_elastic_damp_C(){

	
	int minid;
	calculate_vp_min(&minid);

	// float rr=0.000001;
	float rr=0.001;      //float rr = 0.01;
	// float rr=1;
	dampX<<<Grid_Block.grid,Grid_Block.block>>>(d_damp,pml,&vp[minid-1],vp,nxpml,nzpml,dx,rr);
	dampZ<<<Grid_Block.grid,Grid_Block.block>>>(d_dampz,pml,&vp[minid-1],vp,nxpml,nzpml,dz,rr);

//	elastic_dampX<<<Grid_Block.grid,Grid_Block.block>>>(d_damp,pml,nxpml,nzpml);
//	elastic_dampZ<<<Grid_Block.grid,Grid_Block.block>>>(d_damp,pml,nxpml,nzpml);

	GPUcalculate_elastic_rho_C<<<Grid_Block.grid,Grid_Block.block>>>(d_rho,vp,vs,lamda,miu,nxpml,nzpml);

}

void FD2DGPU_ELASTIC::GPUbufferVPVS(float *vp_all,float *vs_all){

		cudaMemcpy(vel,vp_all,allnx*allnz*sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(vel_s,vs_all,allnx*allnz*sizeof(float),cudaMemcpyHostToDevice);
}


void FD2DGPU_ELASTIC::GPUbufferM(float *pp,float *ps){

		cudaMemcpy(image_pp_m,pp,allnx*allnz*sizeof(float),cudaMemcpyHostToDevice);
		cudaMemcpy(image_ps_m,ps,allnx*allnz*sizeof(float),cudaMemcpyHostToDevice);
}

void FD2DGPU_ELASTIC::buffer_image_extrapolation(const int ngx_min){
	copyvelocityHtoD<<<Grid_Block.grid,Grid_Block.block>>>(image_pp_m,s_image_pp_m,pml,nzpml,ngx_min,allnz,nx,nz);	
	copyvelocityHtoD<<<Grid_Block.grid,Grid_Block.block>>>(image_ps_m,s_image_ps_m,pml,nzpml,ngx_min,allnz,nx,nz);	

	copypmlVz<<<Grid_Block.grid,Grid_Block.block>>>(s_image_pp_m,pml,nxpml,nzpml);
	copypmlVx<<<Grid_Block.grid,Grid_Block.block>>>(s_image_pp_m,pml,nxpml,nzpml);
	copypmlVz<<<Grid_Block.grid,Grid_Block.block>>>(s_image_ps_m,pml,nxpml,nzpml);
	copypmlVx<<<Grid_Block.grid,Grid_Block.block>>>(s_image_ps_m,pml,nxpml,nzpml);	

}


void FD2DGPU_ELASTIC::bufferVpVsHtoD(const int ngx_min){

	copyvelocityHtoD<<<Grid_Block.grid,Grid_Block.block>>>(vel,vp,pml,nzpml,ngx_min,allnz,nx,nz);
	copyvelocityHtoD<<<Grid_Block.grid,Grid_Block.block>>>(vel_s,vs,pml,nzpml,ngx_min,allnz,nx,nz);

	copypmlVz<<<Grid_Block.grid,Grid_Block.block>>>(vp,pml,nxpml,nzpml);
	copypmlVx<<<Grid_Block.grid,Grid_Block.block>>>(vp,pml,nxpml,nzpml);
	copypmlVz<<<Grid_Block.grid,Grid_Block.block>>>(vs,pml,nxpml,nzpml);
	copypmlVx<<<Grid_Block.grid,Grid_Block.block>>>(vs,pml,nxpml,nzpml);

//	cudaMemcpy(vp_bk,vp,nxpml*nzpml*sizeof(float),cudaMemcpyDeviceToDevice);
//	cudaMemcpy(vs_bk,vs,nxpml*nzpml*sizeof(float),cudaMemcpyDeviceToDevice);
}

FD2DGPU_ELASTIC::FD2DGPU_ELASTIC(const float* sou,const float dx1, const float dz1, const float dt1, const int nxpml1, const int nzpml1, \
const int allnx1,const int allnz1,const int scale1,const int pml1,const int nt1,const int nop1,int ntr1):
FD2DGPU(sou,dx1,dz1,dt1,nxpml1,nzpml1,allnx1,allnz1,scale1,pml1,nt1,nop1){
// record related
    ntr = ntr1;
	adds_bac.grid.x = (ntr+BLOCKDIMX-1)/BLOCKDIMX;
	adds_bac.block.x = BLOCKDIMX;	
	cudaMalloc((void**)&GPUrecord,nt*ntr*sizeof(float));
	cudaMalloc((void**)&GPUrecord_x,nt*ntr*sizeof(float));	
	cudaMalloc((void**)&GPUgc,ntr*sizeof(float));	
// 

	int nDim = nxpml*nzpml;
	int nxnz = allnx*allnz;

	cudaMalloc((void**)&txx,nDim*sizeof(float));
	cudaMalloc((void**)&tzz,nDim*sizeof(float));		
	cudaMalloc((void**)&txz,nDim*sizeof(float));
	cudaMalloc((void**)&lamda,nDim*sizeof(float));
	cudaMalloc((void**)&miu,nDim*sizeof(float)); 

	cudaMalloc((void**)&vx_x,nDim*sizeof(float));
	cudaMalloc((void**)&vx_z,nDim*sizeof(float));		
	cudaMalloc((void**)&vz_x,nDim*sizeof(float));
	cudaMalloc((void**)&vz_z,nDim*sizeof(float));
	cudaMalloc((void**)&vp,nDim*sizeof(float));            
	cudaMalloc((void**)&vs,nDim*sizeof(float));  
	cudaMalloc((void**)&vel_s,nxnz*sizeof(float));

	cudaMalloc((void**)&VP,nDim*sizeof(float));            
	cudaMalloc((void**)&VS,nDim*sizeof(float));

//	int ntrace = nx/scale;
//	cudaMalloc((void **)&receiver2,ntrace*nt*sizeof(float));
//	cudaMemset(receiver2,0,ntrace*nt*sizeof(float));			

	cudaMemset(VP,0,nDim*sizeof(float));
    cudaMemset(VS,0,nDim*sizeof(float));
	cudaMemset(vp,0,nDim*sizeof(float));
    cudaMemset(vs,0,nDim*sizeof(float));
	cudaMemset(lamda,0,nDim*sizeof(float));
	cudaMemset(miu,0,nDim*sizeof(float));
	cudaMemset(vel_s,0,nxnz*sizeof(float));

	cudaMalloc((void**)&d_dampz,nDim*sizeof(float));
	cudaMemset(d_dampz,0,nDim*sizeof(float));

	cudaMalloc((void**)&txx_x,nDim*sizeof(float));
	cudaMalloc((void**)&txx_z,nDim*sizeof(float));		
	cudaMalloc((void**)&tzz_x,nDim*sizeof(float));
	cudaMalloc((void**)&tzz_z,nDim*sizeof(float));
	cudaMalloc((void**)&txz_x,nDim*sizeof(float));
	cudaMalloc((void**)&txz_z,nDim*sizeof(float));


//     rotation equation
	vpx = vx_x;
	vpz = vx_z;
	vsx = vz_x;
	vsz = vz_z;
	theta = txx;
	omega = txz;
	theta_x = txx_x;
	theta_z = txx_z;
	omega_x = txz_x;
	omega_z = txz_z;

//
	cudaMalloc((void**)&vp_bk,nDim*sizeof(float));
	cudaMalloc((void**)&vs_bk,nDim*sizeof(float));	
//
	cudaMalloc((void**)&vx_bk,nDim*sizeof(float));
	cudaMalloc((void**)&vz_bk,nDim*sizeof(float));	

	cudaMalloc((void**)&vpx_bk,nDim*sizeof(float));
	cudaMalloc((void**)&vpz_bk,nDim*sizeof(float));		
	cudaMalloc((void**)&vsx_bk,nDim*sizeof(float));
	cudaMalloc((void**)&vsz_bk,nDim*sizeof(float));
	cudaMalloc((void**)&theta_bk,nDim*sizeof(float));
	cudaMalloc((void**)&omega_bk,nDim*sizeof(float));

	cudaMalloc((void**)&theta_x_bk,nDim*sizeof(float));
	cudaMalloc((void**)&theta_z_bk,nDim*sizeof(float));	
	cudaMalloc((void**)&omega_x_bk,nDim*sizeof(float));
	cudaMalloc((void**)&omega_z_bk,nDim*sizeof(float));


//
	vx_x_bk = vpx_bk;
	vx_z_bk = vpz_bk;
	vz_x_bk = vsx_bk;
	vz_z_bk = vsz_bk;
	txx_bk = omega_bk;
	txz_bk = theta_bk;
	cudaMalloc((void**)&tzz_bk,nDim*sizeof(float));
	txx_x_bk = omega_x_bk;
	txx_z_bk = omega_z_bk;
	txz_x_bk = theta_x_bk;
	txz_z_bk = theta_z_bk;

	tzz_x_bk = vp_bk;
	tzz_z_bk = vs_bk;

//



	imagesize.grid.x=(nz+BLOCKDIMX-1)/BLOCKDIMX;
	imagesize.grid.y=(nx+BLOCKDIMY-1)/BLOCKDIMY;
	imagesize.block.x=BLOCKDIMX;
	imagesize.block.y=BLOCKDIMY;

//    rotation equation

// lsrtm 
	cudaMalloc((void**)&s_image_pp_m,nDim*sizeof(float));
	cudaMalloc((void**)&s_image_ps_m,nDim*sizeof(float));
	cudaMalloc((void**)&image_pp_m,nxnz*sizeof(float));
	cudaMalloc((void**)&image_ps_m,nxnz*sizeof(float));
	cudaMalloc((void**)&image_pp,nxnz*sizeof(float));
	cudaMalloc((void**)&image_ps,nxnz*sizeof(float));
	cudaMalloc((void**)&b_theta,nDim*sizeof(float));
	cudaMalloc((void**)&b_omega,nDim*sizeof(float));
	cudaMalloc((void**)&b_theta_x,nDim*sizeof(float));
	cudaMalloc((void**)&b_theta_z,nDim*sizeof(float));
	cudaMalloc((void**)&b_omega_x,nDim*sizeof(float));
	cudaMalloc((void**)&b_omega_z,nDim*sizeof(float));
	cudaMalloc((void**)&b_vpx,nDim*sizeof(float));
	cudaMalloc((void**)&b_vpz,nDim*sizeof(float));
	cudaMalloc((void**)&b_vsx,nDim*sizeof(float));
	cudaMalloc((void**)&b_vsz,nDim*sizeof(float));
	cudaMalloc((void**)&b_vx,nDim*sizeof(float));
	cudaMalloc((void**)&b_vz,nDim*sizeof(float));
	cudaMalloc((void**)&vpx_g,nDim*sizeof(float));
	cudaMalloc((void**)&vpz_g,nDim*sizeof(float));
	cudaMalloc((void**)&vsx_g,nDim*sizeof(float));
	cudaMalloc((void**)&vsz_g,nDim*sizeof(float));	
	cudaMalloc((void**)&GPUrecord_scatter_z,nt*ntr*sizeof(float));
	cudaMalloc((void**)&GPUrecord_scatter_x,nt*ntr*sizeof(float));		
	// cudaMalloc((void**)&numerator_pp,nx*nz*sizeof(float));
	// cudaMalloc((void**)&numerator_ps,nx*nz*sizeof(float));
	// cudaMalloc((void**)&denominator_pp,nx*nz*sizeof(float));
	cudaMalloc((void**)&allimageGPU_numerator_pp,nxnz*sizeof(float));
	cudaMalloc((void**)&allimageGPU_numerator_ps,nxnz*sizeof(float));
	cudaMalloc((void**)&allimageGPU_denominator,nxnz*sizeof(float));
//
	b_txx = b_theta;
	b_txx_x = b_theta_x;
	b_txx_z = b_theta_z;
	b_txz = b_omega;
	b_txz_x = b_omega_x;
	b_txz_z = b_omega_z;
	b_vx_x = b_vpx;
	b_vx_z = b_vpz;
	b_vz_x = b_vsx;
	b_vz_z = b_vsz;

	cudaMalloc((void**)&b_tzz,nDim*sizeof(float));
	cudaMalloc((void**)&b_tzz_x,nDim*sizeof(float));
	cudaMalloc((void**)&b_tzz_z,nDim*sizeof(float));
//
//////////////////////// 2nd scatter wavefield /////////////////////////////////
	cudaMalloc((void**)&b2_vx,nDim*sizeof(float));
	cudaMalloc((void**)&b2_vx_x,nDim*sizeof(float));
	cudaMalloc((void**)&b2_vx_z,nDim*sizeof(float));

	cudaMalloc((void**)&b2_vz,nDim*sizeof(float));
	cudaMalloc((void**)&b2_vz_x,nDim*sizeof(float));
	cudaMalloc((void**)&b2_vz_z,nDim*sizeof(float));

	cudaMalloc((void**)&b2_txx,nDim*sizeof(float));
	cudaMalloc((void**)&b2_txx_x,nDim*sizeof(float));
	cudaMalloc((void**)&b2_txx_z,nDim*sizeof(float));

	cudaMalloc((void**)&b2_tzz,nDim*sizeof(float));
	cudaMalloc((void**)&b2_tzz_x,nDim*sizeof(float));
	cudaMalloc((void**)&b2_tzz_z,nDim*sizeof(float));

	cudaMalloc((void**)&b2_txz,nDim*sizeof(float));
	cudaMalloc((void**)&b2_txz_x,nDim*sizeof(float));
	cudaMalloc((void**)&b2_txz_z,nDim*sizeof(float));

	cudaMalloc((void**)&GPUrecord_scatter2_z,nt*ntr*sizeof(float));
	cudaMalloc((void**)&GPUrecord_scatter2_x,nt*ntr*sizeof(float));	
/////////////////////  receiver-side backward born scatter wavefield//////////
	cudaMalloc((void**)&b_vx_bk,nDim*sizeof(float));
	cudaMalloc((void**)&b_vx_x_bk,nDim*sizeof(float));
	cudaMalloc((void**)&b_vx_z_bk,nDim*sizeof(float));

	cudaMalloc((void**)&b_vz_bk,nDim*sizeof(float));
	cudaMalloc((void**)&b_vz_x_bk,nDim*sizeof(float));
	cudaMalloc((void**)&b_vz_z_bk,nDim*sizeof(float));

	cudaMalloc((void**)&b_txx_bk,nDim*sizeof(float));
	cudaMalloc((void**)&b_txx_x_bk,nDim*sizeof(float));
	cudaMalloc((void**)&b_txx_z_bk,nDim*sizeof(float));

	cudaMalloc((void**)&b_tzz_bk,nDim*sizeof(float));
	cudaMalloc((void**)&b_tzz_x_bk,nDim*sizeof(float));
	cudaMalloc((void**)&b_tzz_z_bk,nDim*sizeof(float));

	cudaMalloc((void**)&b_txz_bk,nDim*sizeof(float));
	cudaMalloc((void**)&b_txz_x_bk,nDim*sizeof(float));
	cudaMalloc((void**)&b_txz_z_bk,nDim*sizeof(float));


	cudaMalloc((void**)&b_vx_gx,nDim*sizeof(float));
	cudaMalloc((void**)&b_vx_gz,nDim*sizeof(float));
	cudaMalloc((void**)&b_vz_gx,nDim*sizeof(float));
	cudaMalloc((void**)&b_vz_gz,nDim*sizeof(float));

	// cudaMalloc((void**)&b_numerator_pp,nx*nz*sizeof(float));
	// cudaMalloc((void**)&b_numerator_ps,nx*nz*sizeof(float));
	// cudaMalloc((void**)&b_denominator_pp,nx*nz*sizeof(float));
	cudaMalloc((void**)&b_allimageGPU_numerator_pp,nxnz*sizeof(float));
	cudaMalloc((void**)&b_allimageGPU_numerator_ps,nxnz*sizeof(float));
	// cudaMalloc((void**)&b_allimageGPU_denominator,nxnz*sizeof(float));

	cudaMemset(b_allimageGPU_numerator_pp,0,nxnz*sizeof(float));
	cudaMemset(b_allimageGPU_numerator_ps,0,nxnz*sizeof(float));
	// cudaMemset(b_allimageGPU_denominator,0,nxnz*sizeof(float));

//
	vx_gx = vpx_g;
	vx_gz = vpz_g;
	vz_gx = vsx_g;
	vz_gz = vsz_g;
//


	cudaMemset(image_pp_m,0,nxnz*sizeof(float));
	cudaMemset(image_ps_m,0,nxnz*sizeof(float));
	cudaMemset(image_pp,0,nxnz*sizeof(float));
	cudaMemset(image_ps,0,nxnz*sizeof(float));	

	cudaMemset(allimageGPU_numerator_pp,0,nxnz*sizeof(float));
	cudaMemset(allimageGPU_numerator_ps,0,nxnz*sizeof(float));
	cudaMemset(allimageGPU_denominator,0,nxnz*sizeof(float));


	// cudaMalloc((void**)&single_pp_grad,nx*nz*sizeof(float));
	// cudaMalloc((void**)&single_ps_grad,nx*nz*sizeof(float));
	cudaMalloc((void**)&pp_gradient,nxnz*sizeof(float));
	cudaMalloc((void**)&ps_gradient,nxnz*sizeof(float));

	cudaMemset(pp_gradient,0,nxnz*sizeof(float));
	cudaMemset(ps_gradient,0,nxnz*sizeof(float));
	
// lsrtm
#ifdef VECTORIMG
	cudaMalloc(&vectorVP,nDim*sizeof(float2));
	cudaMemset(vectorVP,0,nDim*sizeof(float2));
	cudaMalloc(&vectorVS,nDim*sizeof(float2));
	cudaMemset(vectorVS,0,nDim*sizeof(float2));	
	// cudaMalloc(&vector_wavefield,100*nx*nz*sizeof(float2));
	// cpu_vectorVP = new float2[100*nx*nz]{};
	// cudaMalloc(&vector_GPU_FW_wavefield,100*nx*nz*sizeof(float2));
	cudaMalloc(&wavefield,100*nx*nz*sizeof(float2));
	// g++ 4.4.7 not support
//	cpu_wavefield = new float2[100*nx*nz]{};
	// not support
	cpu_wavefield = new float2[100*nx*nz];
	memset(cpu_wavefield,0,sizeof(float2)*100*nx*nz);
	cudaMalloc(&GPU_FW_wavefield,100*nx*nz*sizeof(float2));
#ifdef POYNTING
	cudaMalloc(&wavefield2,100*nx*nz*sizeof(float2));
	cudaMalloc(&GPU_FW_wavefield2,100*nx*nz*sizeof(float2));
	cudaMalloc(&vp_source_up,nDim*sizeof(float2));
	cudaMemset(vp_source_up,0,nDim*sizeof(float2));
	cudaMalloc(&vp_source_down,nDim*sizeof(float2));
	cudaMemset(vp_source_down,0,nDim*sizeof(float2));
	cudaMalloc(&vp_receiver_up,nDim*sizeof(float2));
	cudaMemset(vp_receiver_up,0,nDim*sizeof(float2));
	cudaMalloc(&vp_receiver_down,nDim*sizeof(float2));
	cudaMemset(vp_receiver_down,0,nDim*sizeof(float2));		
#endif
#else
	cudaMalloc(&wavefield,100*nx*nz*sizeof(float));
	cpu_wavefield = new float[100*nx*nz];
	memset(cpu_wavefield,0,100*nx*nz);
	cudaMalloc(&GPU_FW_wavefield,100*nx*nz*sizeof(float));		
#endif

	cudaMalloc(&single_image,nx*nz*sizeof(float));
	cudaMalloc(&single_image_ps,nx*nz*sizeof(float));
	cudaMemset(single_image,0,nx*nz*sizeof(float));	
	cudaMemset(single_image_ps,0,nx*nz*sizeof(float));		
}

FD2DGPU_ELASTIC::~FD2DGPU_ELASTIC(){

	cudaFree(txx);
	cudaFree(tzz);
	cudaFree(txz);
	cudaFree(lamda);
	cudaFree(miu);
	cudaFree(vp);
	cudaFree(vs);
	cudaFree(VP);
	cudaFree(VS);	
	cudaFree(vel_s);
	cudaFree(vx_x);
	cudaFree(vx_z);
	cudaFree(vz_x);
	cudaFree(vz_z);

	cudaFree(d_dampz);
	cudaFree(tzz_x);
	cudaFree(tzz_z);
	cudaFree(txx_x);
	cudaFree(txx_z);
	cudaFree(txz_x);
	cudaFree(txz_z);
	


	cudaFree(wavefield);
	cudaFree(GPU_FW_wavefield);
	delete[] cpu_wavefield;


	cudaFree(GPUrecord);
	cudaFree(GPUrecord_x);	
	cudaFree(GPUgc);


#ifdef VECTORIMG
	cudaFree(vectorVP);
	cudaFree(vectorVS);	
#ifdef POYNTING
	cudaFree(wavefield2);
	cudaFree(GPU_FW_wavefield2);	
	cudaFree(vp_source_up);
	cudaFree(vp_source_down);
	cudaFree(vp_receiver_up);
	cudaFree(vp_receiver_down);			
#endif
#endif

	cudaFree(single_image);	
	cudaFree(single_image_ps);	
//	delete[] cpu_vectorVS;


// image rot-exp //
    cudaFree(theta_bk);
    cudaFree(omega_bk);
    cudaFree(theta_x_bk);
    cudaFree(theta_z_bk);
    cudaFree(omega_x_bk);
    cudaFree(omega_z_bk);
    cudaFree(vpx_bk);
    cudaFree(vpz_bk);
    cudaFree(vsx_bk);
    cudaFree(vsz_bk);
    cudaFree(vx_bk);
    cudaFree(vz_bk);

	cudaFree(vp_bk);
	cudaFree(vs_bk);
//
	cudaFree(tzz_bk);
// lsrtm
    cudaFree(s_image_pp_m);
    cudaFree(s_image_ps_m);
    cudaFree(image_pp_m);
    cudaFree(image_ps_m);
    cudaFree(image_pp);
    cudaFree(image_ps);
    cudaFree(b_theta);
    cudaFree(b_omega);
    cudaFree(b_theta_x);
    cudaFree(b_theta_z);
    cudaFree(b_omega_x);
    cudaFree(b_omega_z);
    cudaFree(b_vpx);
    cudaFree(b_vpz);
    cudaFree(b_vsx);
    cudaFree(b_vsz);
    cudaFree(b_vx);
    cudaFree(b_vz);
    cudaFree(vpx_g);
    cudaFree(vpz_g);
    cudaFree(vsx_g);
    cudaFree(vsz_g);
//
	cudaFree(b_tzz);
	cudaFree(b_tzz_x);
	cudaFree(b_tzz_z);
//
    cudaFree(GPUrecord_scatter_z);
    cudaFree(GPUrecord_scatter_x);
    cudaFree(numerator_pp);
    cudaFree(numerator_ps);
    cudaFree(denominator_pp);
    cudaFree(allimageGPU_numerator_pp);
    cudaFree(allimageGPU_numerator_ps);
    cudaFree(allimageGPU_denominator);

    cudaFree(single_pp_grad);
    cudaFree(single_ps_grad);
    cudaFree(pp_gradient);
    cudaFree(ps_gradient);

	cudaFree(b_vx_bk);
	cudaFree(b_vx_x_bk);
	cudaFree(b_vx_z_bk);		
	cudaFree(b_vz_bk);
	cudaFree(b_vz_x_bk);
	cudaFree(b_vz_z_bk);		
	cudaFree(b_txx_bk);
	cudaFree(b_txx_x_bk);
	cudaFree(b_txx_z_bk);		
	cudaFree(b_tzz_bk);
	cudaFree(b_tzz_x_bk);
	cudaFree(b_tzz_z_bk);		
	cudaFree(b_txz_bk);
	cudaFree(b_txz_x_bk);
	cudaFree(b_txz_z_bk);	

	cudaFree(b_vx_gx);		
	cudaFree(b_vx_gz);
	cudaFree(b_vz_gx);
	cudaFree(b_vz_gz);	

	cudaFree(b2_vx);
	cudaFree(b2_vx_x);
	cudaFree(b2_vx_z);		
	cudaFree(b2_vz);
	cudaFree(b2_vz_x);
	cudaFree(b2_vz_z);		
	cudaFree(b2_txx);
	cudaFree(b2_txx_x);
	cudaFree(b2_txx_z);		
	cudaFree(b2_tzz);
	cudaFree(b2_tzz_x);
	cudaFree(b2_tzz_z);		
	cudaFree(b2_txz);
	cudaFree(b2_txz_x);
	cudaFree(b2_txz_z);

    cudaFree(GPUrecord_scatter2_z);
    cudaFree(GPUrecord_scatter2_x);

    cudaFree(b_numerator_pp);
    cudaFree(b_numerator_ps);
    cudaFree(b_denominator_pp);
    cudaFree(b_allimageGPU_numerator_pp);
    cudaFree(b_allimageGPU_numerator_ps);
    cudaFree(b_allimageGPU_denominator);
// lsrtm
}

__global__ void GPUapplysource(float *p,float *sou,const int isx,const int isz,const int nzpml,const int it)
{
    const int index = blockIdx.x*blockDim.x+threadIdx.x;
	if(index>0)return;
	p[isx*nzpml+isz] +=  sou[it];	
}

void FD2DGPU_ELASTIC::add_virtual_source(){
	born_add_source<<<Grid_Block.grid,Grid_Block.block>>>(b_txx,b_tzz,b_txz,b_txx_x,b_txx_z,b_tzz_x,b_tzz_z,b_txz_x,b_txz_z,vx,vz,s_image_pp_m,s_image_ps_m,lamda,miu,vp,vs,nxpml,nzpml,nop,dx,dz);	
}

void FD2DGPU_ELASTIC::normal_forward_and_born(int direction){

	born_vx<<<Grid_Block.grid,Grid_Block.block>>>(txx,txz,vx,vx_x,vx_z,b_txx,b_txz,b_vx,b_vx_x,b_vx_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);    
	born_vz<<<Grid_Block.grid,Grid_Block.block>>>(tzz,txz,vz,vz_x,vz_z,b_tzz,b_txz,b_vz,b_vz_x,b_vz_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);
	born_txx_tzz<<<Grid_Block.grid,Grid_Block.block>>>(txx,tzz,txx_x,txx_z,tzz_x,tzz_z,vx,vz,b_txx,b_tzz,b_txx_x,b_txx_z,b_tzz_x,b_tzz_z,b_vx,b_vz,s_image_pp_m,s_image_ps_m,vp,vs,\
		nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction);
	born_txz<<<Grid_Block.grid,Grid_Block.block>>>(txz,txz_x,txz_z,vx,vz,b_txz,b_txz_x,b_txz_z,b_vx,b_vz,s_image_ps_m,vs,nxpml,nzpml,dt,dx,dz,nop,miu,d_damp,d_dampz,direction);

}

void FD2DGPU_ELASTIC::born_modeling(int direction,int forward_or_born,int it){

// =1 both forward and born
	if(forward_or_born == 1){

		// update_vx<<<Grid_Block.grid,Grid_Block.block>>>(txx,txz,vx,vx_x,vx_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);    
		// update_vz<<<Grid_Block.grid,Grid_Block.block>>>(tzz,txz,vz,vz_x,vz_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);

		// update_txx_tzz<<<Grid_Block.grid,Grid_Block.block>>>(txx,tzz,txx_x,txx_z,tzz_x,tzz_z,vx,vz,nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction);
		// update_txz<<<Grid_Block.grid,Grid_Block.block>>>(txz,txz_x,txz_z,vx,vz,nxpml,nzpml,dt,dx,dz,nop,miu,d_damp,d_dampz,direction);

		// update_vx<<<Grid_Block.grid,Grid_Block.block>>>(b_txx,b_txz,b_vx,b_vx_x,b_vx_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);    
		// update_vz<<<Grid_Block.grid,Grid_Block.block>>>(b_tzz,b_txz,b_vz,b_vz_x,b_vz_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);

		// add_born_source_components<<<Grid_Block.grid,Grid_Block.block>>>(vx,vz,b_txx_x,b_txx_z,b_tzz_x,b_tzz_z,b_txz_x,b_txz_z,s_image_pp_m,s_image_ps_m,nxpml,nzpml,pml,dx,dz,dt,nop,lamda,miu,d_rho,vp,vs,direction);

		// update_txx_tzz<<<Grid_Block.grid,Grid_Block.block>>>(b_txx,b_tzz,b_txx_x,b_txx_z,b_tzz_x,b_tzz_z,b_vx,b_vz,nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction);
		// update_txz<<<Grid_Block.grid,Grid_Block.block>>>(b_txz,b_txz_x,b_txz_z,b_vx,b_vz,nxpml,nzpml,dt,dx,dz,nop,miu,d_damp,d_dampz,direction);

		update_vx_vz<<<Grid_Block.grid,Grid_Block.block>>>(txx,tzz,txz,vx,vx_x,vx_z,vz,vz_x,vz_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);    
		update_txx_tzz_txz<<<Grid_Block.grid,Grid_Block.block>>>(txx,tzz,txx_x,txx_z,tzz_x,tzz_z,txz,txz_x,txz_z,vx,vz,nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction);

		update_vx_vz<<<Grid_Block.grid,Grid_Block.block>>>(b_txx,b_tzz,b_txz,b_vx,b_vx_x,b_vx_z,b_vz,b_vz_x,b_vz_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);    

		add_born_source_components<<<Grid_Block.grid,Grid_Block.block>>>(vx,vz,b_txx_x,b_txx_z,b_tzz_x,b_tzz_z,b_txz_x,b_txz_z,s_image_pp_m,s_image_ps_m,nxpml,nzpml,pml,dx,dz,dt,nop,lamda,miu,d_rho,vp,vs,direction);

		update_txx_tzz_txz<<<Grid_Block.grid,Grid_Block.block>>>(b_txx,b_tzz,b_txx_x,b_txx_z,b_tzz_x,b_tzz_z,b_txz,b_txz_x,b_txz_z,b_vx,b_vz,nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction);


	}
// // =0 only forward 
// 	if(forward_or_born == 0){
// 		update_vx<<<Grid_Block.grid,Grid_Block.block>>>(txx,txz,vx,vx_x,vx_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);    
// 		update_vz<<<Grid_Block.grid,Grid_Block.block>>>(tzz,txz,vz,vz_x,vz_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);
// 		update_txx_tzz<<<Grid_Block.grid,Grid_Block.block>>>(txx,tzz,txx_x,txx_z,tzz_x,tzz_z,vx,vz,nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction);
// 		update_txz<<<Grid_Block.grid,Grid_Block.block>>>(txz,txz_x,txz_z,vx,vz,nxpml,nzpml,dt,dx,dz,nop,miu,d_damp,d_dampz,direction);
// 	}

// // =3 top part born before low pass
// 	if(forward_or_born == 3){
// 		add_born_source<<<Grid_Block.grid,Grid_Block.block>>>(vx,vz,b_txx,b_tzz,b_txz,s_image_pp_m,s_image_ps_m,nxpml,nzpml,pml,dx,dz,dt,nop,lamda,miu,d_rho,vp,vs,direction);
// 		update_vx<<<Grid_Block.grid,Grid_Block.block>>>(b_txx,b_txz,b_vx,b_vx_x,b_vx_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);    
// 		update_vz<<<Grid_Block.grid,Grid_Block.block>>>(b_tzz,b_txz,b_vz,b_vz_x,b_vz_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);
// 	}
// // =4 bottom part born after low pass
// 	if(forward_or_born == 4){
// 		update_txx_tzz<<<Grid_Block.grid,Grid_Block.block>>>(b_txx,b_tzz,b_txx_x,b_txx_z,b_tzz_x,b_tzz_z,b_vx,b_vz,nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction);
// 		update_txz<<<Grid_Block.grid,Grid_Block.block>>>(b_txz,b_txz_x,b_txz_z,b_vx,b_vz,nxpml,nzpml,dt,dx,dz,nop,miu,d_damp,d_dampz,direction);
// 	}

// // =5 2nd born approximate
// 	if(forward_or_born == 5){
// 		add_born_source<<<Grid_Block.grid,Grid_Block.block>>>(b_vx,b_vz,b2_txx,b2_tzz,b2_txz,s_image_pp_m,s_image_ps_m,nxpml,nzpml,pml,dx,dz,dt,nop,lamda,miu,d_rho,vp,vs,direction);
// 		update_vx<<<Grid_Block.grid,Grid_Block.block>>>(b2_txx,b2_txz,b2_vx,b2_vx_x,b2_vx_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);    
// 		update_vz<<<Grid_Block.grid,Grid_Block.block>>>(b2_tzz,b2_txz,b2_vz,b2_vz_x,b2_vz_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);
// 		update_txx_tzz<<<Grid_Block.grid,Grid_Block.block>>>(b2_txx,b2_tzz,b2_txx_x,b2_txx_z,b2_tzz_x,b2_tzz_z,b2_vx,b2_vz,nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction);
// 		update_txz<<<Grid_Block.grid,Grid_Block.block>>>(b2_txz,b2_txz_x,b2_txz_z,b2_vx,b2_vz,nxpml,nzpml,dt,dx,dz,nop,miu,d_damp,d_dampz,direction);
// 	}

	// =6 reconstruct background wavefield 
	if(forward_or_born == 6 ){
		update_txx_tzz_txz_grad<<<Grid_Block.grid,Grid_Block.block>>>(txx,tzz,txx_x,txx_z,tzz_x,tzz_z,txz,txz_x,txz_z,vx,vz,vx_gx,vz_gz,vx_gz,vz_gx,nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction,pml);
		update_vx_vz<<<Grid_Block.grid,Grid_Block.block>>>(txx,tzz,txz,vx,vx_x,vx_z,vz,vz_x,vz_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);   
	}

	// =8 adjoint wavefield 
	if(forward_or_born == 8 ){
		GPUcalculate_elastic_txx_tzz_txz_back<<<Grid_Block.grid,Grid_Block.block>>>(txx_bk,tzz_bk,txx_x_bk,txx_z_bk,tzz_x_bk,tzz_z_bk,txz_bk,txz_x_bk,txz_z_bk,vx_bk,vz_bk,nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction);	
		GPUcalculate_elastic_vx_vz_back<<<Grid_Block.grid,Grid_Block.block>>>(txx_bk,tzz_bk,txz_bk,vx_bk,vx_x_bk,vx_z_bk,vz_bk,vz_x_bk,vz_z_bk,lamda,miu,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);										
	}

}

// void FD2DGPU_ELASTIC::pure_born_reconstruct_backward(int direction,int forward){

// 		if(forward){
// 			GPUcalculate_elastic_txx_tzz_grad<<<Grid_Block.grid,Grid_Block.block>>>(b_txx,b_tzz,b_txx_x,b_txx_z,b_tzz_x,b_tzz_z,b_vx,b_vz,b_vx_gx,b_vz_gz,\
// 				nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction,pml);
// 			GPUcalculate_elastic_txz_grad<<<Grid_Block.grid,Grid_Block.block>>>(b_txz,b_txz_x,b_txz_z,b_vx,b_vz,b_vx_gz,b_vz_gx,\
// 				nxpml,nzpml,dt,dx,dz,nop,miu,d_damp,d_dampz,direction,pml);		
// 			add_born_source<<<Grid_Block.grid,Grid_Block.block>>>(vx,vz,b_txx,b_tzz,b_txz,s_image_pp_m,s_image_ps_m,nxpml,nzpml,pml,dx,dz,dt,nop,lamda,miu,d_rho,vp,vs,1);				// TODO: if need and -1 or 1
// 			GPUcalculate_elastic_vx<<<Grid_Block.grid,Grid_Block.block>>>(b_txx,b_txz,b_vx,b_vx_x,b_vx_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction,pml);
// 			GPUcalculate_elastic_vz<<<Grid_Block.grid,Grid_Block.block>>>(b_tzz,b_txz,b_vz,b_vz_x,b_vz_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction,pml);

// 			// update_txx_tzz<<<Grid_Block.grid,Grid_Block.block>>>(b_txx,b_tzz,b_txx_x,b_txx_z,b_tzz_x,b_tzz_z,b_vx,b_vz,nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction);
// 			// update_txz<<<Grid_Block.grid,Grid_Block.block>>>(b_txz,b_txz_x,b_txz_z,b_vx,b_vz,nxpml,nzpml,dt,dx,dz,nop,miu,d_damp,d_dampz,direction);	
// 			// update_vx<<<Grid_Block.grid,Grid_Block.block>>>(b_txx,b_txz,b_vx,b_vx_x,b_vx_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);    
// 			// update_vz<<<Grid_Block.grid,Grid_Block.block>>>(b_tzz,b_txz,b_vz,b_vz_x,b_vz_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);
// 			// add_born_source<<<Grid_Block.grid,Grid_Block.block>>>(vx,vz,b_txx,b_tzz,b_txz,s_image_pp_m,s_image_ps_m,nxpml,nzpml,pml,dx,dz,nop,lamda,miu,d_rho,vp,vs);
// 		}
// 		else{		
// 			// add_born_source<<<Grid_Block.grid,Grid_Block.block>>>(vx_bk,vz_bk,b_txx_bk,b_tzz_bk,b_txz_bk,s_image_pp_m,s_image_ps_m,nxpml,nzpml,pml,dx,dz,nop,lamda,miu,d_rho,vp,vs);		
// 			direction = -1;		
// 			GPUcalculate_elastic_vz_back<<<Grid_Block.grid,Grid_Block.block>>>(b_tzz_bk,b_txz_bk,b_vz_bk,b_vz_x_bk,b_vz_z_bk,b_txx_bk,lamda,miu,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);	
// 			GPUcalculate_elastic_vx_back<<<Grid_Block.grid,Grid_Block.block>>>(b_txx_bk,b_txz_bk,b_vx_bk,b_vx_x_bk,b_vx_z_bk,b_tzz_bk,lamda,miu,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);
// 			add_born_source_adjoint<<<Grid_Block.grid,Grid_Block.block>>>(txx_bk,tzz_bk,txz_bk,b_vx_bk,b_vz_bk,s_image_pp_m,s_image_ps_m,nxpml,nzpml,pml,dx,dz,nop,lamda,miu,d_rho,vp,vs,-1);				
// 			GPUcalculate_elastic_txx_tzz<<<Grid_Block.grid,Grid_Block.block>>>(b_txx_bk,b_tzz_bk,b_txx_x_bk,b_txx_z_bk,b_tzz_x_bk,b_tzz_z_bk,b_vx_bk,b_vz_bk,\
// 				nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction);	
// 			GPUcalculate_elastic_txz<<<Grid_Block.grid,Grid_Block.block>>>(b_txz_bk,b_txz_x_bk,b_txz_z_bk,b_vx_bk,b_vz_bk,nxpml,nzpml,dt,dx,dz,nop,miu,d_damp,d_dampz,direction);		

// 			// add_born_source<<<Grid_Block.grid,Grid_Block.block>>>(vx_bk,vz_bk,b_txx_bk,b_tzz_bk,b_txz_bk,s_image_pp_m,s_image_ps_m,nxpml,nzpml,pml,dx,dz,nop,lamda,miu,d_rho,vp,vs,1);  // backward use 1 but reconstruct use -1				
// 			// update_vx<<<Grid_Block.grid,Grid_Block.block>>>(b_txx_bk,b_txz_bk,b_vx_bk,b_vx_x_bk,b_vx_z_bk,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);    
// 			// update_vz<<<Grid_Block.grid,Grid_Block.block>>>(b_tzz_bk,b_txz_bk,b_vz_bk,b_vz_x_bk,b_vz_z_bk,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);	
// 			// update_txx_tzz<<<Grid_Block.grid,Grid_Block.block>>>(b_txx_bk,b_tzz_bk,b_txx_x_bk,b_txx_z_bk,b_tzz_x_bk,b_tzz_z_bk,b_vx_bk,b_vz_bk,nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction);
// 			// update_txz<<<Grid_Block.grid,Grid_Block.block>>>(b_txz_bk,b_txz_x_bk,b_txz_z_bk,b_vx_bk,b_vz_bk,nxpml,nzpml,dt,dx,dz,nop,miu,d_damp,d_dampz,direction);
// 		}		
// }



void FD2DGPU_ELASTIC::calculateVx(int direction,bool forward,bool born){
	if(born){
//			GPUcalculate_elastic_vx<<<Grid_Block.grid,Grid_Block.block>>>(b_txx,b_txz,b_vx,b_vx_x,b_vx_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);
	}
	else{
		if(forward){
			GPUcalculate_elastic_vx<<<Grid_Block.grid,Grid_Block.block>>>(txx,txz,vx,vx_x,vx_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction,pml);
		}
		else{
			GPUcalculate_elastic_vx_back<<<Grid_Block.grid,Grid_Block.block>>>(txx_bk,txz_bk,vx_bk,vx_x_bk,vx_z_bk,tzz_bk,lamda,miu,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);		
		}
	}

}
					
void FD2DGPU_ELASTIC::calculateVz(int direction,bool forward,bool born){
	if(born){
//			GPUcalculate_elastic_vz<<<Grid_Block.grid,Grid_Block.block>>>(b_tzz,b_txz,b_vz,b_vz_x,b_vz_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);
	}
	else{
		if(forward){
			GPUcalculate_elastic_vz<<<Grid_Block.grid,Grid_Block.block>>>(tzz,txz,vz,vz_x,vz_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction,pml);
		}
		else{
			GPUcalculate_elastic_vz_back<<<Grid_Block.grid,Grid_Block.block>>>(tzz_bk,txz_bk,vz_bk,vz_x_bk,vz_z_bk,txx_bk,lamda,miu,nxpml,nzpml,dt,dx,dz,nop,d_rho,d_damp,d_dampz,direction);		
		}		
	}
}
void FD2DGPU_ELASTIC::calculateTxxzz(int direction,bool forward,bool born){
	if(born){
			GPUcalculate_elastic_txx_tzz_born<<<Grid_Block.grid,Grid_Block.block>>>(b_txx,b_tzz,b_txx_x,b_txx_z,b_tzz_x,b_tzz_z,b_vx,b_vz,\
				nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction);		
	}
	else{
		if(forward){
			GPUcalculate_elastic_txx_tzz_grad<<<Grid_Block.grid,Grid_Block.block>>>(txx,tzz,txx_x,txx_z,tzz_x,tzz_z,vx,vz,vx_gx,vz_gz,\
				nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction,pml);
		}
		else{
			GPUcalculate_elastic_txx_tzz<<<Grid_Block.grid,Grid_Block.block>>>(txx_bk,tzz_bk,txx_x_bk,txx_z_bk,tzz_x_bk,tzz_z_bk,vx_bk,vz_bk,\
				nxpml,nzpml,dt,dx,dz,nop,lamda,miu,d_damp,d_dampz,direction);		
		}
	}

}
void FD2DGPU_ELASTIC::calculateTxz(int direction,bool forward,bool born){
	if(born){
			GPUcalculate_elastic_txz_born<<<Grid_Block.grid,Grid_Block.block>>>(b_txz,b_txz_x,b_txz_z,b_vx,b_vz,\
				nxpml,nzpml,dt,dx,dz,nop,miu,d_damp,d_dampz,direction);	
	}
	else{
		if(forward){	
			GPUcalculate_elastic_txz_grad<<<Grid_Block.grid,Grid_Block.block>>>(txz,txz_x,txz_z,vx,vz,vx_gz,vz_gx,\
				nxpml,nzpml,dt,dx,dz,nop,miu,d_damp,d_dampz,direction,pml);
		}
		else{
			GPUcalculate_elastic_txz<<<Grid_Block.grid,Grid_Block.block>>>(txz_bk,txz_x_bk,txz_z_bk,vx_bk,vz_bk,\
				nxpml,nzpml,dt,dx,dz,nop,miu,d_damp,d_dampz,direction);		
		}
	}

}
__global__ void GPUcalculate_helmholtz(float *vx,float *vz,float *VP,float *VS,const int nop,const float dx,const float dz,const int nxpml,const int nzpml){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;

		float tmp_vx_x = 0;
		float tmp_vz_z = 0;
		float tmp_vx_z = 0;
		float tmp_vz_x = 0;		
#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_vx_x += coeff2[i]*(vx[(ix+i-1)*nzpml+iz]-vx[(ix-i)*nzpml+iz]);						//calculate VP
			tmp_vz_z += coeff2[i]*(vz[ix*nzpml+(iz+i-1)]-vz[ix*nzpml+(iz-i)]);						//

			tmp_vx_z += coeff2[i]*(vx[ix*nzpml+(iz+i-1)]-vx[ix*nzpml+(iz-i)]);						//calculate VS	
			tmp_vz_x += coeff2[i]*(vz[(ix+i-1)*nzpml+iz]-vz[(ix-i)*nzpml+iz]);						// 		
		}
		
		VP[iz+ix*nzpml] = tmp_vx_x/dx+tmp_vz_z/dz;
		VS[iz+ix*nzpml] = tmp_vx_z/dz-tmp_vz_x/dx;		

}


__global__ void GPUcalculate_vector_VPVS(float2 *vectorVP,float2 *vectorVS,float *VP,float *VS,const int nop,const float dx,const float dz,const int nxpml,const int nzpml){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;

		float tmp_VP_x = 0;
		float tmp_VP_z = 0;
		float tmp_VS_z = 0;
		float tmp_VS_x = 0;		
#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_VP_x += coeff2[i]*(VP[(ix+i-1)*nzpml+iz]-VP[(ix-i)*nzpml+iz]);						//calculate VP
			tmp_VP_z += coeff2[i]*(VP[ix*nzpml+(iz+i-1)]-VP[ix*nzpml+(iz-i)]);						//

			tmp_VS_z += coeff2[i]*(VS[ix*nzpml+(iz+i-1)]-VS[ix*nzpml+(iz-i)]);						//calculate VS	
			tmp_VS_x += coeff2[i]*(VS[(ix+i-1)*nzpml+iz]-VS[(ix-i)*nzpml+iz]);						// 		
		}
		
		vectorVP[iz+ix*nzpml].x = tmp_VP_x/dx;
		vectorVP[iz+ix*nzpml].y = tmp_VP_z/dz;

		vectorVS[iz+ix*nzpml].x = -tmp_VS_z/dz;		
		vectorVS[iz+ix*nzpml].y = tmp_VS_x/dx;	
}






void FD2DGPU_ELASTIC::Helmholtz_VP_VS(){
	GPUcalculate_helmholtz<<<Grid_Block.grid,Grid_Block.block>>>(vx,vz,VP,VS,nop,dx,dz,nxpml,nzpml);
}

#ifdef VECTORIMG
void FD2DGPU_ELASTIC::Vector_VP_VS(){
		GPUcalculate_vector_VPVS<<<Grid_Block.grid,Grid_Block.block>>>(vectorVP,vectorVS,VP,VS,nop,dx,dz,nxpml,nzpml);
}
#endif

void FD2DGPU_ELASTIC::elastic_addS(const int sxnum,const int it){
   GPUapplysource<<<1,1>>>(txx,source,sxnum,nxpml,nzpml,dt,dx,it,pml,d_rho);		  
   GPUapplysource<<<1,1>>>(tzz,source,sxnum,nxpml,nzpml,dt,dx,it,pml,d_rho);
//   GPUapplysource<<<1,1>>>(vx,source,sxnum,nxpml,nzpml,dt,dx,it,pml,d_rho);
}

void FD2DGPU_ELASTIC::elastic_rotation_addS(const int sxnum,const int it){
   GPUapplysource<<<1,1>>>(theta,source,sxnum,nxpml,nzpml,dt,dx,it,pml,d_rho);		  
}

void FD2DGPU_ELASTIC::elastic_rotation_addS_for(const int it,cudaStream_t stream){
   GPUapplysource<<<1,1,0,stream>>>(txx,source,isx,isz,nzpml,it);	
   GPUapplysource<<<1,1,0,stream>>>(tzz,source,isx,isz,nzpml,it);	   	  
}


void FD2DGPU_ELASTIC::record_elastic(const int scale,const int pml,const int it,const int gy){

     GPUseisrecord<<<record.grid,record.block>>>(vz,vx,receiver,receiver2,it,nt,nx,scale,pml,nzpml,gy);
//    GPUseisrecord<<<record.grid,record.block>>>(VS,receiver,it,nt,nx,scale,pml,nzpml);
}
#ifdef VECTORIMG
void FD2DGPU_ELASTIC::record_elastic_VP_VS(const int scale,const int pml,const int it){

    // GPUseisrecord<<<record.grid,record.block>>>(vz,receiver,it,nt,nx,scale,pml,nzpml);
    GPUseisrecord_float2<<<record.grid,record.block>>>(vectorVP,receiver,it,nt,nx,scale,pml,nzpml);		
}
#endif



__global__ void GPUsnapcopy(float* GPUsnap,float2* vectorVP,float2* vectorVS,const int nxpml,const int nzpml){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-1||ix>nxpml-1)return;
	int index = ix*nzpml+iz;
	int nDim = nxpml*nzpml;
	GPUsnap[index] = vectorVP[index].x;	
	GPUsnap[nDim+index] = vectorVP[index].y;	
	GPUsnap[2*nDim+index] = vectorVS[index].x;	
	GPUsnap[3*nDim+index] = vectorVS[index].y;			
}


// void FD2DGPU_ELASTIC::snapcopy(struct snap snapVPVS,const int nDim){


// 	float* GPUsnap;
// 	cudaMalloc(&GPUsnap,4*nDim*sizeof(float));

// 	GPUsnapcopy<<<Grid_Block.grid,Grid_Block.block>>>(GPUsnap,vectorVP,vectorVS,nxpml,nzpml);

// 	cudaMemcpy(snapVPVS.VP_X,GPUsnap,nDim*sizeof(float),cudaMemcpyDeviceToHost);
// 	cudaMemcpy(snapVPVS.VP_Z,GPUsnap+nDim,nDim*sizeof(float),cudaMemcpyDeviceToHost);
// 	cudaMemcpy(snapVPVS.VS_X,GPUsnap+2*nDim,nDim*sizeof(float),cudaMemcpyDeviceToHost);
// 	cudaMemcpy(snapVPVS.VS_Z,GPUsnap+3*nDim,nDim*sizeof(float),cudaMemcpyDeviceToHost);		

// 	cudaFree(GPUsnap);

			
// }
template<typename T>
__global__ void copyVPtowavefield(T* wavefield,T* VP,const int nxpml,const int nzpml,const int nx,const int nz,const int pml,const int it){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz<pml||ix<pml||iz>nzpml-pml-1||ix>nxpml-pml-1)return;
	wavefield[it*nx*nz+(ix-pml)*nz+(iz-pml)]=VP[ix*nzpml+iz];
}

template<typename T>
void writetodisk(T* wavefield,const int nx,const int nz,const int i,const int id,const int direction,char* tmp_dir){

	FILE *fp = NULL;
	char fn[1024];
	sprintf(fn,"%s/%d_%d_%d_%d_%d_forward.dat",tmp_dir,id,i,direction,nx,nz);
	fp = fopen(fn,"wb");
	fwrite(wavefield,100*nx*nz*sizeof(T),1,fp);
	fclose(fp);
}

template<typename T>
void readfromdisk(T* wavefield,const int nx,const int nz,const int i,const int id,const int direction,char* tmp_dir){

	FILE *fp = NULL;
	char fn[1024];
	sprintf(fn,"%s/%d_%d_%d_%d_%d_forward.dat",tmp_dir,id,i,direction,nx,nz);
	fp = fopen(fn,"rb");
	fread(wavefield,100*nx*nz*sizeof(T),1,fp);
	fclose(fp);
}


__global__ void GPUapplysource_backward(float *vz,float *sou,const int nzpml,const int it,const int pml,const int nx,const int scale,const int nt)
{
//  const int index = (blockIdx.y*gridDim.x)*blockDim.x*blockDim.y+blockIdx.x*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x;
    const int index = blockIdx.x*blockDim.x+threadIdx.x;
	if(index>nx-1)return;

	vz[(index+pml)*nzpml+pml] =  sou[(index*scale*nt)+it];	
}

__device__ float dot(float2 in1, float2 in2) { return in1.x * in2.x + in1.y * in2.y; }


template<typename T>
__global__ void doimage(float *single_image,T *fw,T *bw,const int nx,const int nz,const int nit){
	int iz = blockIdx.x*blockDim.x+threadIdx.x;
	int ix = blockIdx.y*blockDim.y+threadIdx.y;
//	int iz = threadIdx.x;
//	int ix = blockIdx.x;
	if(ix>nx-1||iz>nz-1)return;
	float epsilon=1e-6;
	int i;
	for(i=0;i<nit;i++){
		// float illumination = powf(fw[i*nx*nz+ix*nz+iz],2)+epsilon;
	
#ifdef VECTORIMG
		float illumination = dot(fw[i*nx*nz+ix*nz+iz],fw[i*nx*nz+ix*nz+iz])+epsilon;	
		float cross_correlation = dot(fw[i*nx*nz+ix*nz+iz],bw[i*nx*nz+ix*nz+iz]);
#else
		float illumination = fw[i*nx*nz+ix*nz+iz]*fw[i*nx*nz+ix*nz+iz]+epsilon;	
		float cross_correlation = fw[i*nx*nz+ix*nz+iz]*bw[i*nx*nz+ix*nz+iz];
#endif		
//		single_image[ix*nz+iz] += cross_correlation;
		single_image[ix*nz+iz] += cross_correlation/illumination;
	}
}

__global__ void doimage_poynting(float *single_image,float2 *fw_up,float2* fw_down,float2 *bw_up,float2* bw_down,const int nx,const int nz,const int nit){
	int iz = blockIdx.x*blockDim.x+threadIdx.x;
	int ix = blockIdx.y*blockDim.y+threadIdx.y;
//	int iz = threadIdx.x;
//	int ix = blockIdx.x;
	if(ix>nx-1||iz>nz-1)return;
	float epsilon=1e-6;
	int i;
	for(i=0;i<nit;i++){
		// float illumination = powf(fw[i*nx*nz+ix*nz+iz],2)+epsilon;
	
		float illumination = dot(fw_up[i*nx*nz+ix*nz+iz],fw_up[i*nx*nz+ix*nz+iz])+dot(fw_down[i*nx*nz+ix*nz+iz],fw_down[i*nx*nz+ix*nz+iz])+epsilon;	
		float cross_correlation = dot(fw_up[i*nx*nz+ix*nz+iz],bw_down[i*nx*nz+ix*nz+iz])+dot(fw_down[i*nx*nz+ix*nz+iz],bw_up[i*nx*nz+ix*nz+iz]);
		
//		single_image[ix*nz+iz] += cross_correlation;
		single_image[ix*nz+iz] += cross_correlation/illumination;
	}
}






void FD2DGPU_ELASTIC::elastic_addS_backward(const int it){

	dim3 gridsize((nx+BLOCKDIMX-1)/BLOCKDIMX,1);
	dim3 blocksize(BLOCKDIMX,1);

    GPUapplysource_backward<<<gridsize,blocksize>>>(txx,receiver,nzpml,it,pml,nx,scale,nt);		  
    GPUapplysource_backward<<<gridsize,blocksize>>>(tzz,receiver,nzpml,it,pml,nx,scale,nt);

}

__global__ void poynting(float* txz,float* tzz,float2* vp,float* poynt,const int nxpml,const int nzpml)
{
	int iz = blockIdx.x*blockDim.x+threadIdx.x;
	int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(ix>nxpml-1||iz>nzpml-1)return;
	poynt[ix*nzpml+iz] = -txz[ix*nzpml+iz]*vp[ix*nzpml+iz].x - tzz[ix*nzpml+iz]*vp[ix*nzpml+iz].y;
	vp[ix*nzpml+iz].y = poynt[ix*nzpml+iz]>=0 ? vp[ix*nzpml+iz].y : 0;				// up 
//	vp[ix*nzpml+iz].y = poynt[ix*nzpml+iz]>=0 ? 0 : vp[ix*nzpml+iz].y;				// down
}
__global__ void poynting(float* txz,float* tzz,float2* vp,float* vx,float *vz,float* poynt,const int nxpml,const int nzpml,bool downwave)
{
	int iz = blockIdx.x*blockDim.x+threadIdx.x;
	int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(ix>nxpml-1||iz>nzpml-1)return;
	float2 zerovalue = make_float2(0.0f,0.0f);
	poynt[ix*nzpml+iz] = -txz[ix*nzpml+iz]*vx[ix*nzpml+iz] - tzz[ix*nzpml+iz]*vz[ix*nzpml+iz];
	if(downwave){
		vp[ix*nzpml+iz] = poynt[ix*nzpml+iz]>=0 ? vp[ix*nzpml+iz] : zerovalue;				// down 
	}
	else{
		vp[ix*nzpml+iz] = poynt[ix*nzpml+iz]>=0 ? zerovalue : vp[ix*nzpml+iz];				// up		
	}
//	vp[ix*nzpml+iz].y = poynt[ix*nzpml+iz]>=0 ? 0 : vp[ix*nzpml+iz].y;				// up
}

__global__ void poynting_both_up_down(float* txz,float* tzz,float2* vp,float2* vp_up,float2* vp_down,float* vx,float *vz,float* poynt,const int nxpml,const int nzpml,bool downwave)
{
	int iz = blockIdx.x*blockDim.x+threadIdx.x;
	int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(ix>nxpml-1||iz>nzpml-1)return;
	float2 zerovalue = make_float2(0.0f,0.0f);
	poynt[ix*nzpml+iz] = -txz[ix*nzpml+iz]*vx[ix*nzpml+iz] - tzz[ix*nzpml+iz]*vz[ix*nzpml+iz];

	vp_down[ix*nzpml+iz] = poynt[ix*nzpml+iz]>=0 ? vp[ix*nzpml+iz] : zerovalue;	
	vp_up[ix*nzpml+iz] = poynt[ix*nzpml+iz]>=0 ? zerovalue : vp[ix*nzpml+iz];	
	
//	vp[ix*nzpml+iz].y = poynt[ix*nzpml+iz]>=0 ? 0 : vp[ix*nzpml+iz].y;				// up
}




/*
void FD_ELASTIC(FD2DGPU_ELASTIC &fdtdgpu,const int sxnum,float *rec,const int nDim,float *cpu_single_image,const int myid,char* tmp_dir,char* image_dir,const int gx){

	fdtdgpu.FD_ELASTIC_initial(nDim);
	// cudaStream_t streams[2];
	// for (int i = 0; i < 2; i++) {
  	// cudaStreamCreate(&streams[i]);
	// }

	fdtdgpu.calculate_elastic_damp_C();				// using vt present vp

//	g++ 4.4.7 not support
//	float* bkp = new float[nDim]{0};
//
	float* bkp = new float[nDim];
	memset(bkp,0,sizeof(float)*nDim);

#ifdef SNAP
	struct snap{

		float* VP_X;
		float* VP_Z;
		float* VS_X;
		float* VS_Z;

	}snapVPVS;

	 snapVPVS.VP_X = new float[nDim];
	 snapVPVS.VP_Z = new float[nDim];
	 snapVPVS.VS_X = new float[nDim];
	 snapVPVS.VS_Z = new float[nDim];		
#endif

	float* poynt;
	cudaMalloc(&poynt,fdtdgpu.nxpml*fdtdgpu.nzpml*sizeof(float));
	cudaMemset(poynt,0,fdtdgpu.nxpml*fdtdgpu.nzpml*sizeof(float));
    int it;
	bool downwave = true;

	for (it=0; it<fdtdgpu.nt; ++it)    
    {

        // fdtdgpu.elastic_addS(sxnum,it);

    	fdtdgpu.calculateVx();
		
    	fdtdgpu.calculateVz(); 
				
	fdtdgpu.Helmholtz_VP_VS();


#ifdef VECTORIMG
		fdtdgpu.Vector_VP_VS();
#ifdef PP
//		poynting<<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.txz,fdtdgpu.tzz,fdtdgpu.vectorVP,fdtdgpu.vx,fdtdgpu.vz,poynt,fdtdgpu.nxpml,fdtdgpu.nzpml,downwave);
#ifdef POYNTING
		poynting_both_up_down<<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.txz,fdtdgpu.tzz,fdtdgpu.vectorVP,\
							fdtdgpu.vp_source_up,fdtdgpu.vp_source_down,fdtdgpu.vx,fdtdgpu.vz,poynt,fdtdgpu.nxpml,fdtdgpu.nzpml,downwave);
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.vp_source_up,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);		
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield2,fdtdgpu.vp_source_down,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);		
#else
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.vectorVP,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);
#endif				
#endif
#ifdef PS		
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.vectorVP,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);		
#endif
#ifdef SS		
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.vectorVS,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);		
#endif
#else
#ifdef PP
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.VP,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);
#endif
#ifdef PS
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.VP,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);
#endif
#ifdef SS
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.VS,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);
#endif
#endif

//#ifndef VECTORIMG
		// copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.VP,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);
		if((it+1)%100==0&&it>0){
//			printf("***forward*** it=%d,it/100=%d\n",it,it/100);
			cudaMemcpy(fdtdgpu.cpu_wavefield,fdtdgpu.wavefield,100*fdtdgpu.nx*fdtdgpu.nz*sizeof(datatype),cudaMemcpyDeviceToHost);
			writetodisk<datatype>(fdtdgpu.cpu_wavefield,fdtdgpu.nx,fdtdgpu.nz,(it+1)/100,myid,0,tmp_dir);	
#ifdef POYNTING
			cudaMemcpy(fdtdgpu.cpu_wavefield,fdtdgpu.wavefield2,100*fdtdgpu.nx*fdtdgpu.nz*sizeof(datatype),cudaMemcpyDeviceToHost);
			writetodisk<datatype>(fdtdgpu.cpu_wavefield,fdtdgpu.nx,fdtdgpu.nz,(it+1)/100,myid,1,tmp_dir);
#endif
		}
//#endif

// #ifdef VECTORIMG
// 		copyVPtowavefield<float2><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.vector_wavefield,fdtdgpu.vectorVP,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);
// 		if(it%100==0&&it>0){
// //			printf("***forward*** it=%d,it/100=%d\n",it,it/100);
// 			cudaMemcpy(fdtdgpu.cpu_vectorVP,fdtdgpu.vector_wavefield,100*fdtdgpu.nx*fdtdgpu.nz*sizeof(float2),cudaMemcpyDeviceToHost);
// 			writetodisk<float2>(fdtdgpu.cpu_vectorVP,fdtdgpu.nx,fdtdgpu.nz,it/100,myid);			
// 		}
// #endif

     	fdtdgpu.elastic_addS(sxnum,it);

    	fdtdgpu.calculateTxxzz();
			
   		fdtdgpu.calculateTxz();
			   
        // fdtdgpu.elastic_addS(sxnum,it);
			
//		fdtdgpu.record_elastic_VP_VS(fdtdgpu.scale,fdtdgpu.pml,it);	
		
		fdtdgpu.record_elastic(fdtdgpu.scale,fdtdgpu.pml,it,gx); //  TODO   record both x & z direction .   gx.depth of receivers in Z

//     	GPUseisrecord_mute<<<fdtdgpu.record.grid,fdtdgpu.record.block>>>(fdtdgpu.vz,fdtdgpu.receiver,it,fdtdgpu.nt,fdtdgpu.nx,fdtdgpu.scale,fdtdgpu.pml,fdtdgpu.nzpml,sxnum,fdtdgpu.dx,1500,5e-4,100);

#ifdef SNAP
		char tmps[1024];
			if(it%500==0){
//				fdtdgpu.snapcopy(snapVPVS,nDim);
				float* GPUsnap;
				cudaMalloc(&GPUsnap,4*nDim*sizeof(float));
				GPUsnapcopy<<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(GPUsnap,fdtdgpu.vectorVP,fdtdgpu.vectorVS,fdtdgpu.nxpml,fdtdgpu.nzpml);
				cudaMemcpy(snapVPVS.VP_X,GPUsnap,nDim*sizeof(float),cudaMemcpyDeviceToHost);
				cudaMemcpy(snapVPVS.VP_Z,GPUsnap+nDim,nDim*sizeof(float),cudaMemcpyDeviceToHost);
				cudaMemcpy(snapVPVS.VS_X,GPUsnap+2*nDim,nDim*sizeof(float),cudaMemcpyDeviceToHost);
				cudaMemcpy(snapVPVS.VS_Z,GPUsnap+3*nDim,nDim*sizeof(float),cudaMemcpyDeviceToHost);		
				sprintf(tmps, "snap/snap_FW_VP_X_it%d_%d_%d.dat",it,fdtdgpu.nxpml,fdtdgpu.nzpml);
				write_1d_float_wb(snapVPVS.VP_X,nDim,tmps);
				sprintf(tmps, "snap/snap_FW_VP_Z_it%d_%d_%d.dat",it,fdtdgpu.nxpml,fdtdgpu.nzpml);
				write_1d_float_wb(snapVPVS.VP_Z,nDim,tmps);
				sprintf(tmps, "snap/snap_FW_VS_X_it%d_%d_%d.dat",it,fdtdgpu.nxpml,fdtdgpu.nzpml);
				write_1d_float_wb(snapVPVS.VS_X,nDim,tmps);
				sprintf(tmps, "snap/snap_FW_VS_Z_it%d_%d_%d.dat",it,fdtdgpu.nxpml,fdtdgpu.nzpml);
				write_1d_float_wb(snapVPVS.VS_Z,nDim,tmps);	
				cudaMemcpy(snapVPVS.VS_Z,poynt,nDim*sizeof(float),cudaMemcpyDeviceToHost);
				sprintf(tmps, "snap/snap_poynting_it%d_%d_%d.dat",it,fdtdgpu.nxpml,fdtdgpu.nzpml);
				write_1d_float_wb(snapVPVS.VS_Z,nDim,tmps);							
				cudaFree(GPUsnap);				
//				cudaMemcpy(p,fdtdgpu.VP,nDim*sizeof(float),cudaMemcpyDeviceToHost);
//				sprintf(tmps, "snap/snap_FW_VP_it%d_%d_%d.dat",it,fdtdgpu.nxpml,fdtdgpu.nzpml);
//				write_1d_float_wb(p,nDim,tmps);
//				cudaMemcpy(p,fdtdgpu.VS,nDim*sizeof(float),cudaMemcpyDeviceToHost);
//				sprintf(tmps, "snap/snap_FW_VS_it%d_%d_%d.dat",it,fdtdgpu.nxpml,fdtdgpu.nzpml);
//				write_1d_float_wb(p,nDim,tmps);																				
			}
#endif

    }

#ifdef SNAP
//	 delete[] p;
	 delete[] snapVPVS.VP_X;
	 delete[] snapVPVS.VP_Z;
	 delete[] snapVPVS.VS_X;
	 delete[] snapVPVS.VS_Z;
#endif

//	delete[] cpu_wavefield;
#ifdef RECORD
	int xtrace = fdtdgpu.nx/fdtdgpu.scale;
    cudaMemcpy(rec,fdtdgpu.receiver,xtrace*fdtdgpu.nt*sizeof(float),cudaMemcpyDeviceToHost);
#endif

//////////////////////////////////////////backward wavefield/////////////////////////////////////////

	fdtdgpu.FD_ELASTIC_initial(nDim);

	cudaMemset(fdtdgpu.VP,0,nDim*sizeof(float));
	cudaMemset(fdtdgpu.VS,0,nDim*sizeof(float));

//	float *bkp = new float[nDim];

	int backward_it;

	// float *single_image = nullptr;
	// cudaMalloc(&single_image,fdtdgpu.nx*fdtdgpu.nz*sizeof(float));
	// float *GPU_FW_wavefield = nullptr;
	// cudaMalloc(&GPU_FW_wavefield,100*fdtdgpu.nx*fdtdgpu.nz*sizeof(float));

//	float *fw_wavefield = new float[100*fdtdgpu.nx*fdtdgpu.nz]{};

//	float *cpu_single_image = new float[fdtdgpu.nx*fdtdgpu.nz];

	dim3 gridimage((fdtdgpu.nz+BLOCKDIMX-1)/BLOCKDIMX,(fdtdgpu.nx+BLOCKDIMY-1)/BLOCKDIMY);
	dim3 blockimage(BLOCKDIMX,BLOCKDIMY);

// #ifndef VECTORIMG
// 	cudaMemset(fdtdgpu.wavefield,0,100*fdtdgpu.nx*fdtdgpu.nz*sizeof(float));
// #endif

// #ifdef VECTORIMG
// 	cudaMemset(fdtdgpu.vector_wavefield,0,100*fdtdgpu.nx*fdtdgpu.nz*sizeof(float2));
// #endif
	cudaMemset(fdtdgpu.wavefield,0,100*fdtdgpu.nx*fdtdgpu.nz*sizeof(datatype));

#ifdef POYNTING
	cudaMemset(fdtdgpu.wavefield2,0,100*fdtdgpu.nx*fdtdgpu.nz*sizeof(datatype));
#endif

	downwave = false;

for (it=fdtdgpu.nt-1; it>=0; --it)    
    {
//		backward_it = fdtdgpu.nt-1-it;
		backward_it = it;
        fdtdgpu.elastic_addS_backward(backward_it);

    	fdtdgpu.calculateVx();
		
    	fdtdgpu.calculateVz(); 
				
		fdtdgpu.Helmholtz_VP_VS();

		// if(it%500==0){
		// 	char tmps[1024];
		// 	cudaMemcpy(bkp,fdtdgpu.VP,nDim*sizeof(float),cudaMemcpyDeviceToHost);
		// 	sprintf(tmps, "snap/snap_BW_VP_it%d_%d_%d.dat",it,fdtdgpu.nxpml,fdtdgpu.nzpml);
		// 	write_1d_float_wb(bkp,nDim,tmps);
		// }
#ifdef VECTORIMG		
		fdtdgpu.Vector_VP_VS();
		if(it<fdtdgpu.nt-1)
#ifdef PP
//		poynting<<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.txz,fdtdgpu.tzz,fdtdgpu.vectorVP,fdtdgpu.vx,fdtdgpu.vz,poynt,fdtdgpu.nxpml,fdtdgpu.nzpml,downwave);	
#ifdef POYNTING
		poynting_both_up_down<<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.txz,fdtdgpu.tzz,fdtdgpu.vectorVP,\
							fdtdgpu.vp_receiver_up,fdtdgpu.vp_receiver_down,fdtdgpu.vx,fdtdgpu.vz,poynt,fdtdgpu.nxpml,fdtdgpu.nzpml,downwave);
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.vp_receiver_up,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);		
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield2,fdtdgpu.vp_receiver_down,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);		
#elsel
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.vectorVP,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);	
#endif
#endif
#ifdef PS
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.vectorVS,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);	
#endif
#ifdef SS
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.vectorVS,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);	
#endif
#else
		if(it<fdtdgpu.nt-1)
#ifdef PP		
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.VP,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);			
#endif
#ifdef PS		
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.VS,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);			
#endif
#ifdef SS		
		copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.VS,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);			
#endif
#endif

//#ifndef VECTORIMG
		// copyVPtowavefield<datatype><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.wavefield,fdtdgpu.VP,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);
		if(it%100==0&&it<fdtdgpu.nt-1){
//			printf("***backward*** it=%d,it/100=%d\n",it,it/100);
			readfromdisk<datatype>(fdtdgpu.cpu_wavefield,fdtdgpu.nx,fdtdgpu.nz,(it/100)+1,myid,0,tmp_dir);
			cudaMemcpy(fdtdgpu.GPU_FW_wavefield,fdtdgpu.cpu_wavefield,100*fdtdgpu.nx*fdtdgpu.nz*sizeof(datatype),cudaMemcpyHostToDevice);
#ifdef POYNTING
			readfromdisk<datatype>(fdtdgpu.cpu_wavefield,fdtdgpu.nx,fdtdgpu.nz,(it/100)+1,myid,1,tmp_dir);
			cudaMemcpy(fdtdgpu.GPU_FW_wavefield2,fdtdgpu.cpu_wavefield,100*fdtdgpu.nx*fdtdgpu.nz*sizeof(datatype),cudaMemcpyHostToDevice);
			doimage_poynting<<<gridimage,blockimage>>>(fdtdgpu.single_image,fdtdgpu.GPU_FW_wavefield,fdtdgpu.GPU_FW_wavefield2,fdtdgpu.wavefield,fdtdgpu.wavefield2,fdtdgpu.nx,fdtdgpu.nz,100);
#else
			doimage<datatype><<<gridimage,blockimage>>>(fdtdgpu.single_image,fdtdgpu.GPU_FW_wavefield,fdtdgpu.wavefield,fdtdgpu.nx,fdtdgpu.nz,100);
#endif
		}
//#endif

// #ifdef VECTORIMG
// 		copyVPtowavefield<float2><<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.vector_wavefield,fdtdgpu.vectorVS,fdtdgpu.nxpml,fdtdgpu.nzpml,fdtdgpu.nx,fdtdgpu.nz,fdtdgpu.pml,it%100);
// 		if(it%100==0&&it<fdtdgpu.nt-1){
// //			printf("***backward*** it=%d,it/100=%d\n",it,it/100);
// 			readfromdisk<float2>(fdtdgpu.cpu_vectorVP,fdtdgpu.nx,fdtdgpu.nz,(it/100)+1,myid);
// 			cudaMemcpy(fdtdgpu.vector_GPU_FW_wavefield,fdtdgpu.cpu_vectorVP,100*fdtdgpu.nx*fdtdgpu.nz*sizeof(float2),cudaMemcpyHostToDevice);
// 			doimage<float2><<<gridimage,blockimage>>>(fdtdgpu.single_image,fdtdgpu.vector_GPU_FW_wavefield,fdtdgpu.vector_wavefield,fdtdgpu.nx,fdtdgpu.nz,100);
// 		}
// #endif

    	fdtdgpu.calculateTxxzz();
			
   		fdtdgpu.calculateTxz();
			   
//		fdtdgpu.record_elastic_VP_VS(fdtdgpu.scale,fdtdgpu.pml,it);	
		
    }

	char path[1024];
	cudaMemcpy(cpu_single_image,fdtdgpu.single_image,fdtdgpu.nx*fdtdgpu.nz*sizeof(float),cudaMemcpyDeviceToHost);
	sprintf(path, "%s/single_image_%d_%d_%d.dat",image_dir,sxnum,fdtdgpu.nx,fdtdgpu.nz);
	write_1d_float_wb(cpu_single_image,fdtdgpu.nx*fdtdgpu.nz,path);	

	cudaFree(poynt);
//////////////////////////////////////////backward wavefield/////////////////////////////////////////
//	 delete[] fw_wavefield;
//	 delete[] cpu_single_image;
//	 delete[] bkp;
	// cudaFree(GPU_FW_wavefield);
	// cudaFree(single_image);
//	cudaFree(wavefield);
}
*/

__global__ void GPUcalculate_rotation_v(float *theta,float *omega,float *vx,float *vz,float *vpx,float *vpz,float *vsx,float *vsz,float *vp,float *vs,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,const int nop,float *rho,int direction)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;

		float tmp_theta_x = 0;
		float tmp_theta_z = 0;
		float tmp_omega_x = 0;
		float tmp_omega_z = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_theta_x += coeff2[i]*(theta[(ix+i-1)*nzpml+iz]-theta[(ix-i)*nzpml+iz]);
			tmp_theta_z += coeff2[i]*(theta[ix*nzpml+(iz+i-1)]-theta[ix*nzpml+(iz-i)]);
			tmp_omega_x += coeff2[i]*(omega[(ix+i)*nzpml+iz]-omega[(ix-i+1)*nzpml+iz]);
			tmp_omega_z += coeff2[i]*(omega[ix*nzpml+(iz+i)]-omega[ix*nzpml+(iz-i+1)]);
		}
		
		vpx[iz+ix*nzpml] = vpx[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dx)*tmp_theta_x;
		vpz[iz+ix*nzpml] = vpz[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dz)*tmp_theta_z;

		vsx[iz+ix*nzpml] = vsx[iz+ix*nzpml]+direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dz)*tmp_omega_z;
		vsz[iz+ix*nzpml] = vsz[iz+ix*nzpml]-direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dx)*tmp_omega_x;

		vx[iz+ix*nzpml] = vpx[iz+ix*nzpml] + vsx[iz+ix*nzpml];
		vz[iz+ix*nzpml] = vpz[iz+ix*nzpml] + vsz[iz+ix*nzpml];

}

__global__ void GPUcalculate_rotation_p(float *theta,float *omega,float *vx,float *vz,float *theta_x,float *theta_z,float *omega_x,float *omega_z,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,const int nop,float *rho,int direction)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;

		float tmp_vx_x = 0;
		float tmp_vx_z = 0;
		float tmp_vz_x = 0;
		float tmp_vz_z = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_vx_x += coeff2[i]*(vx[(ix+i)*nzpml+iz]-vx[(ix-i+1)*nzpml+iz]);
			tmp_vx_z += coeff2[i]*(vx[ix*nzpml+(iz+i-1)]-vx[ix*nzpml+(iz-i)]);
			tmp_vz_x += coeff2[i]*(vz[(ix+i-1)*nzpml+iz]-vz[(ix-i)*nzpml+iz]);
			tmp_vz_z += coeff2[i]*(vz[ix*nzpml+(iz+i)]-vz[ix*nzpml+(iz-i+1)]);
		}
	
		theta_x[iz+ix*nzpml] = theta_x[iz+ix*nzpml]+direction*(dt/dx)*tmp_vx_x;
		theta_z[iz+ix*nzpml] = theta_z[iz+ix*nzpml]+direction*(dt/dz)*tmp_vz_z;

		omega_x[iz+ix*nzpml] = omega_x[iz+ix*nzpml]-direction*(dt/dx)*tmp_vz_x;
		omega_z[iz+ix*nzpml] = omega_z[iz+ix*nzpml]+direction*(dt/dz)*tmp_vx_z;

		theta[iz+ix*nzpml] = theta_x[iz+ix*nzpml] + theta_z[iz+ix*nzpml];
		omega[iz+ix*nzpml] = omega_x[iz+ix*nzpml] + omega_z[iz+ix*nzpml];

}


/////////////////////////// LSRTM  releated re-construct wave field/////////////////////////////////////////
__global__ void GPUcalculate_rotation_v_reconstruct(float *theta,float *omega,float *vx,float *vz,float *vpx,float *vpz,float *vsx,float *vsz,float *vp,float *vs,const int nxpml, const int nzpml,const float dt,\
	const float dx,const float dz,const int nop,float *rho,int direction,float *vpx_g,float *vpz_g,float *vsx_g,float *vsz_g)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;

		float tmp_theta_x = 0;
		float tmp_theta_z = 0;
		float tmp_omega_x = 0;
		float tmp_omega_z = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_theta_x += coeff2[i]*(theta[(ix+i-1)*nzpml+iz]-theta[(ix-i)*nzpml+iz]);
			tmp_theta_z += coeff2[i]*(theta[ix*nzpml+(iz+i-1)]-theta[ix*nzpml+(iz-i)]);
			tmp_omega_x += coeff2[i]*(omega[(ix+i)*nzpml+iz]-omega[(ix-i+1)*nzpml+iz]);
			tmp_omega_z += coeff2[i]*(omega[ix*nzpml+(iz+i)]-omega[ix*nzpml+(iz-i+1)]);
		}
		
		vpx[iz+ix*nzpml] = vpx[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dx)*tmp_theta_x;
		vpz[iz+ix*nzpml] = vpz[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dz)*tmp_theta_z;

		vsx[iz+ix*nzpml] = vsx[iz+ix*nzpml]+direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dz)*tmp_omega_z;
		vsz[iz+ix*nzpml] = vsz[iz+ix*nzpml]-direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dx)*tmp_omega_x;

		vx[iz+ix*nzpml] = vpx[iz+ix*nzpml] + vsx[iz+ix*nzpml];
		vz[iz+ix*nzpml] = vpz[iz+ix*nzpml] + vsz[iz+ix*nzpml];

////     used to calculate cross-correalation/gradient //
		vpx_g[iz+ix*nzpml] = (vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(1.0/dx)*tmp_theta_x;
		vpz_g[iz+ix*nzpml] = (vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(1.0/dz)*tmp_theta_z;
		vsx_g[iz+ix*nzpml] = (vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(1.0/dx)*tmp_omega_z;
		vsz_g[iz+ix*nzpml] = -(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(1.0/dz)*tmp_omega_x;		
////
}


//reload reconstruction using pml boundary
__global__ void GPUcalculate_rotation_v_reconstruct(float *theta,float *omega,float *vx,float *vz,float *vpx,float *vpz,float *vsx,float *vsz,float *vp,float *vs,const int nxpml, const int nzpml,const float dt,\
	const float dx,const float dz,const int nop,float *rho,int direction,float *vpx_g,float *vpz_g,float *vsx_g,float *vsz_g,float *dampx,float *dampz)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;

		float damp1 = 1 - dt*dampx[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*dampx[iz+ix*nzpml]/2;
		float damp3 = 1 - dt*dampz[iz+ix*nzpml]/2;
		float damp4 = 1 + dt*dampz[iz+ix*nzpml]/2;

		float tmp_theta_x = 0;
		float tmp_theta_z = 0;
		float tmp_omega_x = 0;
		float tmp_omega_z = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_theta_x += coeff2[i]*(theta[(ix+i-1)*nzpml+iz]-theta[(ix-i)*nzpml+iz]);
			tmp_theta_z += coeff2[i]*(theta[ix*nzpml+(iz+i-1)]-theta[ix*nzpml+(iz-i)]);
			tmp_omega_x += coeff2[i]*(omega[(ix+i)*nzpml+iz]-omega[(ix-i+1)*nzpml+iz]);
			tmp_omega_z += coeff2[i]*(omega[ix*nzpml+(iz+i)]-omega[ix*nzpml+(iz-i+1)]);
		}
		
		float vpx_next = vpx[iz+ix*nzpml];
		float vpz_next = vpx[iz+ix*nzpml];
		float vsx_next = vpx[iz+ix*nzpml];		
		float vsz_next = vpx[iz+ix*nzpml];

		vpx[iz+ix*nzpml] = (damp1*vpx[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dx)*tmp_theta_x)/damp2;
		vpz[iz+ix*nzpml] = (damp3*vpz[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dz)*tmp_theta_z)/damp4;

		vsx[iz+ix*nzpml] = (damp1*vsx[iz+ix*nzpml]+direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dz)*tmp_omega_z)/damp2;
		vsz[iz+ix*nzpml] = (damp3*vsz[iz+ix*nzpml]-direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dx)*tmp_omega_x)/damp4;

		vx[iz+ix*nzpml] = vpx[iz+ix*nzpml] + vsx[iz+ix*nzpml];
		vz[iz+ix*nzpml] = vpz[iz+ix*nzpml] + vsz[iz+ix*nzpml];

////     used to calculate cross-correalation/gradient //
/*
		vpx_g[iz+ix*nzpml] = (vp[iz+ix*nzpml]*2)*(dt/dx)*tmp_theta_x;					// dt/dx instead of 1.0/dx
		vpz_g[iz+ix*nzpml] = (vp[iz+ix*nzpml]*2)*(dt/dz)*tmp_theta_z;
		vsx_g[iz+ix*nzpml] = (vs[iz+ix*nzpml]*2)*(dt/dz)*tmp_omega_z;						//tmp_omega_x
		vsz_g[iz+ix*nzpml] = -(vs[iz+ix*nzpml]*2)*(dt/dx)*tmp_omega_x;		
*/

//		vpx_g[iz+ix*nzpml] = (vp[iz+ix*nzpml]*2)*((1.0/dx)*tmp_theta_x + (1.0/dz)*tmp_theta_z);	
//		vsx_g[iz+ix*nzpml] = -(vs[iz+ix*nzpml]*2)*((1.0/dz)*tmp_omega_z + (1.0/dx)*tmp_omega_x);


/*			maybe right
		vpx_g[iz+ix*nzpml] = (2 * vp[iz+ix*nzpml] / (vp[iz+ix*nzpml]*vp[iz+ix*nzpml])) * (vpx[iz+ix*nzpml] - vpx_before) / dt;
		vpz_g[iz+ix*nzpml] = (2 * vp[iz+ix*nzpml] / (vp[iz+ix*nzpml]*vp[iz+ix*nzpml])) * (vpz[iz+ix*nzpml] - vpz_before) / dt;

		vsx_g[iz+ix*nzpml] = -(2 * vs[iz+ix*nzpml] / (vs[iz+ix*nzpml]*vs[iz+ix*nzpml])) * (vsx[iz+ix*nzpml] - vsx_before) / dt;
		vsz_g[iz+ix*nzpml] = -(2 * vs[iz+ix*nzpml] / (vs[iz+ix*nzpml]*vs[iz+ix*nzpml])) * (vsz[iz+ix*nzpml] - vsz_before) / dt;
*/

/*
		vpx_g[iz+ix*nzpml] = - (2 * vp[iz+ix*nzpml] / (vp[iz+ix*nzpml]*vp[iz+ix*nzpml])) * (vpx_next - vpx[iz+ix*nzpml]) / 1;
		vpz_g[iz+ix*nzpml] = - (2 * vp[iz+ix*nzpml] / (vp[iz+ix*nzpml]*vp[iz+ix*nzpml])) * (vpz_next - vpz[iz+ix*nzpml]) / 1;

		vsx_g[iz+ix*nzpml] =   (2 * vs[iz+ix*nzpml] / (vs[iz+ix*nzpml]*vs[iz+ix*nzpml])) * (vsx_next - vsx[iz+ix*nzpml]) / 1;
		vsz_g[iz+ix*nzpml] =   (2 * vs[iz+ix*nzpml] / (vs[iz+ix*nzpml]*vs[iz+ix*nzpml])) * (vsz_next - vsz[iz+ix*nzpml]) / 1;
*/

		vpx_g[iz+ix*nzpml] = (vpx_next - vpx[iz+ix*nzpml]) / dt /(vp[iz+ix*nzpml]*vp[iz+ix*nzpml]);
		vpz_g[iz+ix*nzpml] = (vpz_next - vpz[iz+ix*nzpml]) / dt /(vp[iz+ix*nzpml]*vp[iz+ix*nzpml]);

		vsx_g[iz+ix*nzpml] = -(vsx_next - vsx[iz+ix*nzpml]) / dt /(vs[iz+ix*nzpml]*vs[iz+ix*nzpml]);
		vsz_g[iz+ix*nzpml] = -(vsz_next - vsz[iz+ix*nzpml]) / dt/ (vs[iz+ix*nzpml]*vs[iz+ix*nzpml]);


}
//


//////////////////////////////// LSRTM  releated //////////////////////////////////////////////
__global__ void GPUcalculate_rotation_v_born(float *theta,float *omega,float *vx,float *vz,float *vpx,float *vpz,float *vsx,float *vsz,float *vp,float *vs, \
const int nxpml, const int nzpml,const float dt,const float dx,const float dz,const int nop,float *rho,int direction,\
float *b_theta,float *b_omega,float *b_vpx,float *b_vpz,float *b_vsx,float *b_vsz,float *b_vx,float *b_vz,float *image_pp,float *image_ps)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;

		float tmp_theta_x = 0;
		float tmp_theta_z = 0;
		float tmp_omega_x = 0;
		float tmp_omega_z = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_theta_x += coeff2[i]*(theta[(ix+i-1)*nzpml+iz]-theta[(ix-i)*nzpml+iz]);
			tmp_theta_z += coeff2[i]*(theta[ix*nzpml+(iz+i-1)]-theta[ix*nzpml+(iz-i)]);
			tmp_omega_x += coeff2[i]*(omega[(ix+i)*nzpml+iz]-omega[(ix-i+1)*nzpml+iz]);
			tmp_omega_z += coeff2[i]*(omega[ix*nzpml+(iz+i)]-omega[ix*nzpml+(iz-i+1)]);
		}
		
		vpx[iz+ix*nzpml] = vpx[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dx)*tmp_theta_x;
		vpz[iz+ix*nzpml] = vpz[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dz)*tmp_theta_z;

		vsx[iz+ix*nzpml] = vsx[iz+ix*nzpml]+direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dz)*tmp_omega_z;
		vsz[iz+ix*nzpml] = vsz[iz+ix*nzpml]-direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dx)*tmp_omega_x;

		vx[iz+ix*nzpml] = vpx[iz+ix*nzpml] + vsx[iz+ix*nzpml];
		vz[iz+ix*nzpml] = vpz[iz+ix*nzpml] + vsz[iz+ix*nzpml];

//	born scatter wavefield
//  add virtual source to make scatter wave
		b_theta[iz+ix*nzpml] += image_pp[iz+ix*nzpml]*(tmp_theta_x/dx+tmp_theta_z/dz)*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml]);
		b_omega[iz+ix*nzpml] += image_ps[iz+ix*nzpml]*(tmp_omega_z/dx-tmp_omega_x/dz)*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml]);
//
		tmp_theta_x = 0;
		tmp_theta_z = 0;
		tmp_omega_x = 0;
		tmp_omega_z = 0;
#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_theta_x += coeff2[i]*(b_theta[(ix+i-1)*nzpml+iz]-b_theta[(ix-i)*nzpml+iz]);
			tmp_theta_z += coeff2[i]*(b_theta[ix*nzpml+(iz+i-1)]-b_theta[ix*nzpml+(iz-i)]);
			tmp_omega_x += coeff2[i]*(b_omega[(ix+i)*nzpml+iz]-b_omega[(ix-i+1)*nzpml+iz]);
			tmp_omega_z += coeff2[i]*(b_omega[ix*nzpml+(iz+i)]-b_omega[ix*nzpml+(iz-i+1)]);
		}
		b_vpx[iz+ix*nzpml] = b_vpx[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dx)*tmp_theta_x;
		b_vpz[iz+ix*nzpml] = b_vpz[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dz)*tmp_theta_z;

		b_vsx[iz+ix*nzpml] = b_vsx[iz+ix*nzpml]+direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dz)*tmp_omega_z;
		b_vsz[iz+ix*nzpml] = b_vsz[iz+ix*nzpml]-direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dx)*tmp_omega_x;

		b_vx[iz+ix*nzpml] = b_vpx[iz+ix*nzpml] + b_vsx[iz+ix*nzpml];
		b_vz[iz+ix*nzpml] = b_vpz[iz+ix*nzpml] + b_vsz[iz+ix*nzpml];
//
}

//reload v_born function using pml boundary
__global__ void GPUcalculate_rotation_v_born(float *theta,float *omega,float *vx,float *vz,float *vpx,float *vpz,float *vsx,float *vsz,float *vp,float *vs, \
const int nxpml, const int nzpml,const float dt,const float dx,const float dz,const int nop,float *rho,int direction,\
float *b_theta,float *b_omega,float *b_vpx,float *b_vpz,float *b_vsx,float *b_vsz,float *b_vx,float *b_vz,float *image_pp,float *image_ps,float *dampx,float *dampz)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;

		float damp1 = 1 - dt*dampx[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*dampx[iz+ix*nzpml]/2;
		float damp3 = 1 - dt*dampz[iz+ix*nzpml]/2;
		float damp4 = 1 + dt*dampz[iz+ix*nzpml]/2;

		float tmp_theta_x = 0;
		float tmp_theta_z = 0;
		float tmp_omega_x = 0;
		float tmp_omega_z = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_theta_x += coeff2[i]*(theta[(ix+i-1)*nzpml+iz]-theta[(ix-i)*nzpml+iz]);
			tmp_theta_z += coeff2[i]*(theta[ix*nzpml+(iz+i-1)]-theta[ix*nzpml+(iz-i)]);
			tmp_omega_x += coeff2[i]*(omega[(ix+i)*nzpml+iz]-omega[(ix-i+1)*nzpml+iz]);
			tmp_omega_z += coeff2[i]*(omega[ix*nzpml+(iz+i)]-omega[ix*nzpml+(iz-i+1)]);
		}
		
		float vpx_before = vpx[iz+ix*nzpml];
		float vpz_before = vpz[iz+ix*nzpml];
		float vsx_before = vsx[iz+ix*nzpml];
		float vsz_before = vsz[iz+ix*nzpml];

		vpx[iz+ix*nzpml] = (damp1*vpx[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dx)*tmp_theta_x)/damp2;
		vpz[iz+ix*nzpml] = (damp3*vpz[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dz)*tmp_theta_z)/damp4;

		vsx[iz+ix*nzpml] = (damp1*vsx[iz+ix*nzpml]+direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dz)*tmp_omega_z)/damp2;		//TODO:  /dx   or  /dz
		vsz[iz+ix*nzpml] = (damp3*vsz[iz+ix*nzpml]-direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dx)*tmp_omega_x)/damp4;

		vx[iz+ix*nzpml] = vpx[iz+ix*nzpml] + vsx[iz+ix*nzpml];
		vz[iz+ix*nzpml] = vpz[iz+ix*nzpml] + vsz[iz+ix*nzpml];

//	born scatter wavefield
//  add virtual source to make scatter wave
		// b_theta[iz+ix*nzpml] += image_pp[iz+ix*nzpml]*(tmp_theta_x/dx+tmp_theta_z/dz)*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml]);			//TODO   range of image.   nx*nz or  nxpml*nzpml
		// b_omega[iz+ix*nzpml] += image_ps[iz+ix*nzpml]*(tmp_omega_z/dx-tmp_omega_x/dz)*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml]);
// add virtual source to vp & vs
/*
		b_vpx[iz+ix*nzpml] += image_pp[iz+ix*nzpml]*(tmp_theta_x/dx)*(vp[iz+ix*nzpml]*2);
		b_vpz[iz+ix*nzpml] += image_pp[iz+ix*nzpml]*(tmp_theta_z/dz)*(vp[iz+ix*nzpml]*2);
		b_vsx[iz+ix*nzpml] += image_ps[iz+ix*nzpml]*(tmp_omega_z/dz)*(vs[iz+ix*nzpml]*2);					// TODO:  confirm
		b_vsz[iz+ix*nzpml] += image_ps[iz+ix*nzpml]*(-tmp_omega_x/dx)*(vs[iz+ix*nzpml]*2);			
*/

		b_vpx[iz+ix*nzpml] += image_pp[iz+ix*nzpml]*2*(vpx[iz+ix*nzpml] - vpx_before)/dt;
		b_vpz[iz+ix*nzpml] += image_pp[iz+ix*nzpml]*2*(vpz[iz+ix*nzpml] - vpz_before)/dt;
		b_vsx[iz+ix*nzpml] += image_ps[iz+ix*nzpml]*2*(vsx[iz+ix*nzpml] - vsx_before)/dt;
		b_vsz[iz+ix*nzpml] += image_ps[iz+ix*nzpml]*2*(vsz[iz+ix*nzpml] - vsz_before)/dt;

		tmp_theta_x = 0;
		tmp_theta_z = 0;
		tmp_omega_x = 0;
		tmp_omega_z = 0;
#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_theta_x += coeff2[i]*(b_theta[(ix+i-1)*nzpml+iz]-b_theta[(ix-i)*nzpml+iz]);
			tmp_theta_z += coeff2[i]*(b_theta[ix*nzpml+(iz+i-1)]-b_theta[ix*nzpml+(iz-i)]);
			tmp_omega_x += coeff2[i]*(b_omega[(ix+i)*nzpml+iz]-b_omega[(ix-i+1)*nzpml+iz]);
			tmp_omega_z += coeff2[i]*(b_omega[ix*nzpml+(iz+i)]-b_omega[ix*nzpml+(iz-i+1)]);
		}
		b_vpx[iz+ix*nzpml] = (damp1*b_vpx[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dx)*tmp_theta_x)/damp2;				//TODO whether need pml 
		b_vpz[iz+ix*nzpml] = (damp3*b_vpz[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dz)*tmp_theta_z)/damp4;

		b_vsx[iz+ix*nzpml] = (damp1*b_vsx[iz+ix*nzpml]+direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dz)*tmp_omega_z)/damp2;
		b_vsz[iz+ix*nzpml] = (damp3*b_vsz[iz+ix*nzpml]-direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dx)*tmp_omega_x)/damp4;

		b_vx[iz+ix*nzpml] = b_vpx[iz+ix*nzpml] + b_vsx[iz+ix*nzpml];
		b_vz[iz+ix*nzpml] = b_vpz[iz+ix*nzpml] + b_vsz[iz+ix*nzpml];

}
//



__global__ void GPUcalculate_rotation_p_born(float *theta,float *omega,float *vx,float *vz,float *theta_x,float *theta_z,float *omega_x,float *omega_z, \
const int nxpml, const int nzpml,const float dt,const float dx,const float dz,const int nop,float *rho,int direction,\
float *b_vx,float *b_vz,float *b_theta_x,float *b_theta_z,float *b_omega_x,float *b_omega_z,float *b_theta,float *b_omega)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;

		float tmp_vx_x = 0;
		float tmp_vx_z = 0;
		float tmp_vz_x = 0;
		float tmp_vz_z = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_vx_x += coeff2[i]*(vx[(ix+i)*nzpml+iz]-vx[(ix-i+1)*nzpml+iz]);
			tmp_vx_z += coeff2[i]*(vx[ix*nzpml+(iz+i-1)]-vx[ix*nzpml+(iz-i)]);
			tmp_vz_x += coeff2[i]*(vz[(ix+i-1)*nzpml+iz]-vz[(ix-i)*nzpml+iz]);
			tmp_vz_z += coeff2[i]*(vz[ix*nzpml+(iz+i)]-vz[ix*nzpml+(iz-i+1)]);
		}
	
		theta_x[iz+ix*nzpml] = theta_x[iz+ix*nzpml]+direction*(dt/dx)*tmp_vx_x;
		theta_z[iz+ix*nzpml] = theta_z[iz+ix*nzpml]+direction*(dt/dz)*tmp_vz_z;

		omega_x[iz+ix*nzpml] = omega_x[iz+ix*nzpml]-direction*(dt/dx)*tmp_vz_x;
		omega_z[iz+ix*nzpml] = omega_z[iz+ix*nzpml]+direction*(dt/dz)*tmp_vx_z;

		theta[iz+ix*nzpml] = theta_x[iz+ix*nzpml] + theta_z[iz+ix*nzpml];
		omega[iz+ix*nzpml] = omega_x[iz+ix*nzpml] + omega_z[iz+ix*nzpml];
		
////////////// ////////////////born scatter wave-field  //////////////////////////////
		tmp_vx_x = 0;
		tmp_vx_z = 0;
		tmp_vz_x = 0;
		tmp_vz_z = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_vx_x += coeff2[i]*(b_vx[(ix+i)*nzpml+iz]-b_vx[(ix-i+1)*nzpml+iz]);
			tmp_vx_z += coeff2[i]*(b_vx[ix*nzpml+(iz+i-1)]-b_vx[ix*nzpml+(iz-i)]);
			tmp_vz_x += coeff2[i]*(b_vz[(ix+i-1)*nzpml+iz]-b_vz[(ix-i)*nzpml+iz]);
			tmp_vz_z += coeff2[i]*(b_vz[ix*nzpml+(iz+i)]-b_vz[ix*nzpml+(iz-i+1)]);
		}
	
		b_theta_x[iz+ix*nzpml] = b_theta_x[iz+ix*nzpml]+direction*(dt/dx)*tmp_vx_x;
		b_theta_z[iz+ix*nzpml] = b_theta_z[iz+ix*nzpml]+direction*(dt/dz)*tmp_vz_z;

		b_omega_x[iz+ix*nzpml] = b_omega_x[iz+ix*nzpml]-direction*(dt/dx)*tmp_vz_x;
		b_omega_z[iz+ix*nzpml] = b_omega_z[iz+ix*nzpml]+direction*(dt/dz)*tmp_vx_z;

		b_theta[iz+ix*nzpml] = b_theta_x[iz+ix*nzpml] + b_theta_z[iz+ix*nzpml];
		b_omega[iz+ix*nzpml] = b_omega_x[iz+ix*nzpml] + b_omega_z[iz+ix*nzpml];
///////////////////////
}


//reload p_born function using pml boundary
__global__ void GPUcalculate_rotation_p_born(float *theta,float *omega,float *vx,float *vz,float *theta_x,float *theta_z,float *omega_x,float *omega_z, \
const int nxpml, const int nzpml,const float dt,const float dx,const float dz,const int nop,float *rho,int direction,\
float *b_vx,float *b_vz,float *b_theta_x,float *b_theta_z,float *b_omega_x,float *b_omega_z,float *b_theta,float *b_omega,float *dampx,float *dampz)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;


		float damp1 = 1 - dt*dampx[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*dampx[iz+ix*nzpml]/2;
		float damp3 = 1 - dt*dampz[iz+ix*nzpml]/2;
		float damp4 = 1 + dt*dampz[iz+ix*nzpml]/2;


		float tmp_vx_x = 0;
		float tmp_vx_z = 0;
		float tmp_vz_x = 0;
		float tmp_vz_z = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_vx_x += coeff2[i]*(vx[(ix+i)*nzpml+iz]-vx[(ix-i+1)*nzpml+iz]);
			tmp_vx_z += coeff2[i]*(vx[ix*nzpml+(iz+i-1)]-vx[ix*nzpml+(iz-i)]);
			tmp_vz_x += coeff2[i]*(vz[(ix+i-1)*nzpml+iz]-vz[(ix-i)*nzpml+iz]);
			tmp_vz_z += coeff2[i]*(vz[ix*nzpml+(iz+i)]-vz[ix*nzpml+(iz-i+1)]);
		}
	
		theta_x[iz+ix*nzpml] = (damp1*theta_x[iz+ix*nzpml]+direction*(dt/dx)*tmp_vx_x)/damp2;
		theta_z[iz+ix*nzpml] = (damp3*theta_z[iz+ix*nzpml]+direction*(dt/dz)*tmp_vz_z)/damp4;

		omega_x[iz+ix*nzpml] = (damp1*omega_x[iz+ix*nzpml]-direction*(dt/dx)*tmp_vz_x)/damp2;
		omega_z[iz+ix*nzpml] = (damp3*omega_z[iz+ix*nzpml]+direction*(dt/dz)*tmp_vx_z)/damp4;

		theta[iz+ix*nzpml] = theta_x[iz+ix*nzpml] + theta_z[iz+ix*nzpml];
		omega[iz+ix*nzpml] = omega_x[iz+ix*nzpml] + omega_z[iz+ix*nzpml];
		
////////////// ////////////////born scatter wave-field  //////////////////////////////
		tmp_vx_x = 0;
		tmp_vx_z = 0;
		tmp_vz_x = 0;
		tmp_vz_z = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_vx_x += coeff2[i]*(b_vx[(ix+i)*nzpml+iz]-b_vx[(ix-i+1)*nzpml+iz]);
			tmp_vx_z += coeff2[i]*(b_vx[ix*nzpml+(iz+i-1)]-b_vx[ix*nzpml+(iz-i)]);
			tmp_vz_x += coeff2[i]*(b_vz[(ix+i-1)*nzpml+iz]-b_vz[(ix-i)*nzpml+iz]);
			tmp_vz_z += coeff2[i]*(b_vz[ix*nzpml+(iz+i)]-b_vz[ix*nzpml+(iz-i+1)]);
		}
	
		b_theta_x[iz+ix*nzpml] = (damp1*b_theta_x[iz+ix*nzpml]+direction*(dt/dx)*tmp_vx_x)/damp2;
		b_theta_z[iz+ix*nzpml] = (damp3*b_theta_z[iz+ix*nzpml]+direction*(dt/dz)*tmp_vz_z)/damp4;

		b_omega_x[iz+ix*nzpml] = (damp1*b_omega_x[iz+ix*nzpml]-direction*(dt/dx)*tmp_vz_x)/damp2;
		b_omega_z[iz+ix*nzpml] = (damp3*b_omega_z[iz+ix*nzpml]+direction*(dt/dz)*tmp_vx_z)/damp4;

		b_theta[iz+ix*nzpml] = b_theta_x[iz+ix*nzpml] + b_theta_z[iz+ix*nzpml];
		b_omega[iz+ix*nzpml] = b_omega_x[iz+ix*nzpml] + b_omega_z[iz+ix*nzpml];
}
//


__global__ void GPUcalculate_rotation_v(float *theta,float *omega,float *vx,float *vz,float *vpx,float *vpz,float *vsx,float *vsz,float *vp,float *vs,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,const int nop,float *rho,int direction,float *dampx,float *dampz)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;
//	if(iz>=nzpml-nop||ix>=nxpml-nop||iz<nop||ix<nop)return;
//	__syncthreads();
		float damp1 = 1 - dt*dampx[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*dampx[iz+ix*nzpml]/2;
		float damp3 = 1 - dt*dampz[iz+ix*nzpml]/2;
		float damp4 = 1 + dt*dampz[iz+ix*nzpml]/2;

		float tmp_theta_x = 0;
		float tmp_theta_z = 0;
		float tmp_omega_x = 0;
		float tmp_omega_z = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_theta_x += coeff2[i]*(theta[(ix+i-1)*nzpml+iz]-theta[(ix-i)*nzpml+iz]);
			tmp_theta_z += coeff2[i]*(theta[ix*nzpml+(iz+i-1)]-theta[ix*nzpml+(iz-i)]);
			tmp_omega_x += coeff2[i]*(omega[(ix+i)*nzpml+iz]-omega[(ix-i+1)*nzpml+iz]);
			tmp_omega_z += coeff2[i]*(omega[ix*nzpml+(iz+i)]-omega[ix*nzpml+(iz-i+1)]);
		}
		
		vpx[iz+ix*nzpml] = (damp1*vpx[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dx)*tmp_theta_x)/damp2;
		vpz[iz+ix*nzpml] = (damp3*vpz[iz+ix*nzpml]+direction*(vp[iz+ix*nzpml]*vp[iz+ix*nzpml])*(dt/dz)*tmp_theta_z)/damp4;

		vsx[iz+ix*nzpml] = (damp1*vsx[iz+ix*nzpml]+direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dz)*tmp_omega_z)/damp2;
		vsz[iz+ix*nzpml] = (damp3*vsz[iz+ix*nzpml]-direction*(vs[iz+ix*nzpml]*vs[iz+ix*nzpml])*(dt/dx)*tmp_omega_x)/damp4;

		vx[iz+ix*nzpml] = vpx[iz+ix*nzpml] + vsx[iz+ix*nzpml];
		vz[iz+ix*nzpml] = vpz[iz+ix*nzpml] + vsz[iz+ix*nzpml];

}

__global__ void GPUcalculate_rotation_p(float *theta,float *omega,float *vx,float *vz,float *theta_x,float *theta_z,float *omega_x,float *omega_z,const int nxpml, const int nzpml,const float dt,const float dx,const float dz,const int nop,float *rho,int direction,float *dampx,float *dampz)
{

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz>nzpml-nop||ix>nxpml-nop||iz<nop||ix<nop)return;
//	if(iz>=nzpml-nop||ix>=nxpml-nop||iz<nop||ix<nop)return;

//	__syncthreads();
		float damp1 = 1 - dt*dampx[iz+ix*nzpml]/2;
		float damp2 = 1 + dt*dampx[iz+ix*nzpml]/2;
		float damp3 = 1 - dt*dampz[iz+ix*nzpml]/2;
		float damp4 = 1 + dt*dampz[iz+ix*nzpml]/2;

		float tmp_vx_x = 0;
		float tmp_vx_z = 0;
		float tmp_vz_x = 0;
		float tmp_vz_z = 0;

#pragma unroll 4
		for(int i=1;i<=nop;i++)
		{
			tmp_vx_x += coeff2[i]*(vx[(ix+i)*nzpml+iz]-vx[(ix-i+1)*nzpml+iz]);
			tmp_vx_z += coeff2[i]*(vx[ix*nzpml+(iz+i-1)]-vx[ix*nzpml+(iz-i)]);
			tmp_vz_x += coeff2[i]*(vz[(ix+i-1)*nzpml+iz]-vz[(ix-i)*nzpml+iz]);
			tmp_vz_z += coeff2[i]*(vz[ix*nzpml+(iz+i)]-vz[ix*nzpml+(iz-i+1)]);
		}
	
		theta_x[iz+ix*nzpml] = (damp1*theta_x[iz+ix*nzpml]+direction*(dt/dx)*tmp_vx_x)/damp2;
		theta_z[iz+ix*nzpml] = (damp3*theta_z[iz+ix*nzpml]+direction*(dt/dz)*tmp_vz_z)/damp4;

		omega_x[iz+ix*nzpml] = (damp1*omega_x[iz+ix*nzpml]-direction*(dt/dx)*tmp_vz_x)/damp2;
		omega_z[iz+ix*nzpml] = (damp3*omega_z[iz+ix*nzpml]+direction*(dt/dz)*tmp_vx_z)/damp4;

		theta[iz+ix*nzpml] = theta_x[iz+ix*nzpml] + theta_z[iz+ix*nzpml];
		omega[iz+ix*nzpml] = omega_x[iz+ix*nzpml] + omega_z[iz+ix*nzpml];
		
}

__global__ void stack_image(float *numerator_pp,float *numerator_ps,float *denominator,float *allimageGPU_numerator_pp,float *allimageGPU_numerator_ps,float *allimageGPU_denominator,const int nx,const int nz,const int ngx_left){
	int iz = blockIdx.x*blockDim.x+threadIdx.x;
	int ix = blockIdx.y*blockDim.y+threadIdx.y;

	if(ix>nx-1||iz>nz-1)return;
	allimageGPU_numerator_pp[(ngx_left+ix)*nz+iz] += numerator_pp[ix*nz+iz];
	allimageGPU_numerator_ps[(ngx_left+ix)*nz+iz] += numerator_ps[ix*nz+iz];	
	allimageGPU_denominator[(ngx_left+ix)*nz+iz] += denominator[ix*nz+iz];			
}


__global__ void GPUrecord_scatterwavefield(float *sz,float *sx,float* scatter_record_z,float* scatter_record_x,int* gc,const int it,const int ngx_left,const int pml,const int nzpml,const int ntr,const int dx,const int nt,const int igz)
{
    const int index = blockIdx.x*blockDim.x+threadIdx.x;
	if(index>ntr-1)return;
	int igx = pml + (int)((float)(gc[index]-ngx_left)/dx);
	scatter_record_z[(index*nt)+it] = sz[igx*nzpml+igz];
	scatter_record_x[(index*nt)+it] = sx[igx*nzpml+igz];		
//	scatter_record[(index*nt)+it] = 0;								//  ???????
}

__global__ void GPU_subtract(float *scatter_z,float *scatter_x,float *record_z,float *record_x,const int nt,const int ntr,const float alpha){
    const int it = blockIdx.x*blockDim.x+threadIdx.x;
	const int itrace = blockIdx.y*blockDim.y+threadIdx.y;
	if(it<nt&&itrace<ntr){

		scatter_z[itrace*nt+it] = scatter_z[itrace*nt+it] - record_z[itrace*nt+it];
		scatter_x[itrace*nt+it] = scatter_x[itrace*nt+it] - record_x[itrace*nt+it];

		// scatter_z[itrace*nt+it] = record_z[itrace*nt+it] - scatter_z[itrace*nt+it];
		// scatter_x[itrace*nt+it] = record_x[itrace*nt+it] - scatter_x[itrace*nt+it];
	}
}

__global__ void GPU_subtract_alpha(float *scatter_z,float *scatter_x,float *record_z,float *record_x,const int nt,const int ntr){
    const int it = blockIdx.x*blockDim.x+threadIdx.x;
	const int itrace = blockIdx.y*blockDim.y+threadIdx.y;
	if(it<nt&&itrace<ntr){

		scatter_z[itrace*nt+it] = record_z[itrace*nt+it] - scatter_z[itrace*nt+it];
		scatter_x[itrace*nt+it] = record_x[itrace*nt+it] - scatter_x[itrace*nt+it];
	}
}



__global__ void GPU_add(float *scatter_z,float *scatter_x,float *record_z,float *record_x,const int nt,const int ntr,const float alpha){
    const int it = blockIdx.x*blockDim.x+threadIdx.x;
	const int itrace = blockIdx.y*blockDim.y+threadIdx.y;
	if(it<nt&&itrace<ntr){
		// record_z[itrace*nt+it] = record_z[itrace*nt+it] + alpha*scatter_z[itrace*nt+it];
		// record_x[itrace*nt+it] = record_x[itrace*nt+it] + alpha*scatter_x[itrace*nt+it];


		// scatter_z[itrace*nt+it] -= record_z[itrace*nt+it];
		// scatter_x[itrace*nt+it] -= record_x[itrace*nt+it];	

		scatter_z[itrace*nt+it] = alpha*scatter_z[itrace*nt+it] + record_z[itrace*nt+it];
		scatter_x[itrace*nt+it] = alpha*scatter_x[itrace*nt+it] + record_x[itrace*nt+it];							
	}
}




void FD2DGPU_ELASTIC::subtract(double alpha,bool doublescatter){
	dim3 gridsize((nt+BLOCKDIMX-1)/BLOCKDIMX,(ntr+BLOCKDIMY-1)/BLOCKDIMY);
	dim3 blocksize(BLOCKDIMX,BLOCKDIMY);
	GPU_subtract<<<gridsize,blocksize>>>(GPUrecord_scatter_z,GPUrecord_scatter_x,GPUrecord,GPUrecord_x,nt,ntr,alpha);
	if(doublescatter){
		GPU_add<<<gridsize,blocksize>>>(GPUrecord_scatter_z,GPUrecord_scatter_x,GPUrecord_scatter2_z,GPUrecord_scatter2_x,nt,ntr,alpha);		
	}
}


void FD2DGPU_ELASTIC::subtract_alpha(){
	dim3 gridsize((nt+BLOCKDIMX-1)/BLOCKDIMX,(ntr+BLOCKDIMY-1)/BLOCKDIMY);
	dim3 blocksize(BLOCKDIMX,BLOCKDIMY);
	GPU_subtract_alpha<<<gridsize,blocksize>>>(GPUrecord_scatter_z,GPUrecord_scatter_x,GPUrecord,GPUrecord_x,nt,ntr);

}


void FD2DGPU_ELASTIC::scatter_add(double alpha){
	dim3 gridsize((nt+BLOCKDIMX-1)/BLOCKDIMX,(ntr+BLOCKDIMY-1)/BLOCKDIMY);
	dim3 blocksize(BLOCKDIMX,BLOCKDIMY);

	GPU_add<<<gridsize,blocksize>>>(GPUrecord_scatter_z,GPUrecord_scatter_x,GPUrecord_scatter2_z,GPUrecord_scatter2_x,nt,ntr,alpha);		
}



void FD2DGPU_ELASTIC::set_random_boundary(){
///////////////////////////////////// vp random boundary //////////////////////////////////////
	dim3 block_b(BLOCKDIMX,BLOCKDIMY);
	dim3 grid_b((pml+BLOCKDIMX-1)/BLOCKDIMX,(nxpml+BLOCKDIMY-1)/BLOCKDIMY);
	curandState* devStates;
//	std::cout<<"block = "<< block_b.x <<"\t" << block_b.y <<"  grid = "<< grid_b.x <<"\t"<< grid_b.y <<std::endl;
	int N = block_b.x*block_b.y*grid_b.x*grid_b.y;
//	std::cout<<"N = "<<N<<std::endl;  
	cudaMalloc(&devStates,N	* sizeof(curandState));
	srand(time(0));
	unsigned long seed = rand();	
	int maxid;	
	float *cpu_vmax = new float[1];
	calculate_max(&maxid,vp);
	cudaMemcpy(cpu_vmax,&vp[maxid-1],sizeof(float),cudaMemcpyDeviceToHost);

//	std::cout<<"after max, maxid = "<<maxid<<" max vp = "<<*cpu_vmax<<std::endl;
	*cpu_vmax += 0.0;

	float k;
	k = *cpu_vmax/pml;
//	k = 1.0*dx;	
//	k *= 1.0;	

	setup_kernel<<<grid_b,block_b>>>(devStates,seed,nxpml,pml);
	random_vel<<<grid_b,block_b>>>(devStates,vp,vs,cpu_vmax[0],nxpml,pml,nzpml,k,nxpml,pml,1);

	dim3 block_c(BLOCKDIMX,BLOCKDIMY);
	dim3 grid_c((nzpml+BLOCKDIMX-1)/BLOCKDIMX,(pml+BLOCKDIMY-1)/BLOCKDIMY);
	N = block_c.x*block_c.y*grid_c.x*grid_c.y;
	cudaFree(devStates);
	cudaMalloc(&devStates,N	* sizeof(curandState));
	setup_kernel<<<grid_c,block_c>>>(devStates,seed,pml,nzpml);
	random_vel<<<grid_c,block_c>>>(devStates,vp,vs,cpu_vmax[0],pml,nzpml,nzpml,k,nxpml,pml,0);
	cudaFree(devStates);

	GPUcalculate_elastic_rho_C<<<Grid_Block.grid,Grid_Block.block>>>(d_rho,vp,vs,lamda,miu,nxpml,nzpml);

////////////////////////////		vp random boundary	//////////////////////////////////////
/*
	calculate_max(&maxid,vs);
	cudaMemcpy(cpu_vmax,&vs[maxid-1],sizeof(float),cudaMemcpyDeviceToHost);
//	std::cout<<"after max, maxid = "<<maxid<<" max vs = "<<*cpu_vmax<<std::endl;
	k = *cpu_vmax/pml;

	N = block_b.x*block_b.y*grid_b.x*grid_b.y;
	cudaFree(devStates);
	cudaMalloc(&devStates,N	* sizeof(curandState));
	setup_kernel<<<grid_b,block_b>>>(devStates,seed,nxpml,pml);
	random_vel<<<grid_b,block_b>>>(devStates,vs,cpu_vmax[0],nxpml,pml,nzpml,k,nxpml,pml,1);

	N = block_c.x*block_c.y*grid_c.x*grid_c.y;
	cudaFree(devStates);
	cudaMalloc(&devStates,N	* sizeof(curandState));
	setup_kernel<<<grid_c,block_c>>>(devStates,seed,pml,nzpml);
	random_vel<<<grid_c,block_c>>>(devStates,vs,cpu_vmax[0],pml,nzpml,nzpml,k,nxpml,pml,0);

	cudaFree(devStates);
	delete[] cpu_vmax;
*/
}

__global__ void doimage(float *single_image,float *single_image_ps,float *vpx,float *vpz,float *vpx_bk,float *vpz_bk, \
float *vsx_bk,float *vsz_bk,const int nx,const int nz,const int pml,const int nzpml){
	int iz = blockIdx.x*blockDim.x+threadIdx.x;
	int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(ix>nx-1||iz>nz-1)return;
	float epsilon = 1e-6;
	float pp = vpx[(ix+pml)*nzpml+iz+pml]*vpx_bk[(ix+pml)*nzpml+iz+pml] + vpz[(ix+pml)*nzpml+iz+pml]*vpz_bk[(ix+pml)*nzpml+iz+pml];
	float pp_source = vpx[(ix+pml)*nzpml+iz+pml]*vpx[(ix+pml)*nzpml+iz+pml] + vpz[(ix+pml)*nzpml+iz+pml]*vpz[(ix+pml)*nzpml+iz+pml];
//	single_image[ix*nz+iz] += pp;
	single_image[ix*nz+iz] += pp/(pp_source+epsilon);
//	single_image[(ix+pml)*nzpml+iz+pml] += (vpx[(ix+pml)*nzpml+iz+pml]+vpz[(ix+pml)*nzpml+iz+pml])*(vpx_bk[(ix+pml)*nzpml+iz+pml]+vpz_bk[(ix+pml)*nzpml+iz+pml]);
	
	float ps = vpx[(ix+pml)*nzpml+iz+pml]*vsx_bk[(ix+pml)*nzpml+iz+pml] + vpz[(ix+pml)*nzpml+iz+pml]*vsz_bk[(ix+pml)*nzpml+iz+pml];	
	single_image_ps[ix*nz+iz] += ps/(pp_source+epsilon);

}

__global__ void doimage_illum(float *allimageGPU_denominator,float *vx,float *vz,const int nx,const int nz,const int delta_left,const int record_left_in_v,const int pml,const int nzpml){
	int iz = blockIdx.x*blockDim.x+threadIdx.x;
	int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(ix>nx-1||iz>nz-1)return;
	float illumination = vx[(ix+pml)*nzpml+iz+pml]*vx[(ix+pml)*nzpml+iz+pml]+vz[(ix+pml)*nzpml+iz+pml]*vz[(ix+pml)*nzpml+iz+pml];
	float scale = 1.0;
	allimageGPU_denominator[(delta_left+record_left_in_v+ix)*nz+iz] += scale*illumination;	
}


__global__ void doimage_gradient(float *allimageGPU_numerator_pp,float *allimageGPU_numerator_ps,float *vx_gx,float *vz_gz,float *vx_gz,float *vz_gx,float *txx_bk,float *tzz_bk,float *txz_bk,float *vp,float *vs,float *rho,const int nx,const int nz,const int delta_left,const int record_left_in_v,const int pml,const int nzpml){
	int iz = blockIdx.x*blockDim.x+threadIdx.x;
	int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(ix>nx-1||iz>nz-1)return;

	float scale = 1.0;

// 	float lamda_grad = ((vx_gx[(ix+pml)*nzpml+iz+pml]+vz_gz[(ix+pml)*nzpml+iz+pml])*(txx_bk[(ix+pml)*nzpml+iz+pml]+tzz_bk[(ix+pml)*nzpml+iz+pml]));
// 	float miu_grad = -2*(vx_gx[(ix+pml)*nzpml+iz+pml]*tzz_bk[(ix+pml)*nzpml+iz+pml]+vz_gz[(ix+pml)*nzpml+iz+pml]*txx_bk[(ix+pml)*nzpml+iz+pml]) \
// 					+(vx_gz[(ix+pml)*nzpml+iz+pml]+vz_gx[(ix+pml)*nzpml+iz+pml])*txz_bk[(ix+pml)*nzpml+iz+pml];  //   not real miu_grad . it's been calculated .
// //A new elastic least‑squares reverse‑time migration method based on the new gradient equations



//real lamda_grad and miu_grad 
	float lamda_grad = -((vx_gx[(ix+pml)*nzpml+iz+pml]+vz_gz[(ix+pml)*nzpml+iz+pml])*(txx_bk[(ix+pml)*nzpml+iz+pml]+tzz_bk[(ix+pml)*nzpml+iz+pml]));
	float miu_grad = -2*(vx_gx[(ix+pml)*nzpml+iz+pml]*txx_bk[(ix+pml)*nzpml+iz+pml]+vz_gz[(ix+pml)*nzpml+iz+pml]*tzz_bk[(ix+pml)*nzpml+iz+pml]) \
					-(vx_gz[(ix+pml)*nzpml+iz+pml]+vz_gx[(ix+pml)*nzpml+iz+pml])*txz_bk[(ix+pml)*nzpml+iz+pml];  



// delta_vp delta_vs
	float cross_correlation_pp = 2*vp[(ix+pml)*nzpml+iz+pml]*rho[(ix+pml)*nzpml+iz+pml]*lamda_grad;
	float cross_correlation_ps = -4*vs[(ix+pml)*nzpml+iz+pml]*rho[(ix+pml)*nzpml+iz+pml]*lamda_grad + 2*vs[(ix+pml)*nzpml+iz+pml]*rho[(ix+pml)*nzpml+iz+pml]*miu_grad;


// refl_p  refl_s
	// float cross_correlation_pp = 2*vp[(ix+pml)*nzpml+iz+pml]*vp[(ix+pml)*nzpml+iz+pml]*rho[(ix+pml)*nzpml+iz+pml]*lamda_grad;
	// float cross_correlation_ps = -4*vs[(ix+pml)*nzpml+iz+pml]*vs[(ix+pml)*nzpml+iz+pml]*rho[(ix+pml)*nzpml+iz+pml]*lamda_grad \
	// 							+ 2*vs[(ix+pml)*nzpml+iz+pml]*vs[(ix+pml)*nzpml+iz+pml]*rho[(ix+pml)*nzpml+iz+pml]*miu_grad;


	allimageGPU_numerator_pp[(delta_left+record_left_in_v+ix)*nz+iz] += cross_correlation_pp;
	allimageGPU_numerator_ps[(delta_left+record_left_in_v+ix)*nz+iz] += cross_correlation_ps;	

}


__global__ void doimage_gradient(float *allimageGPU_numerator_pp,float *allimageGPU_numerator_ps,float *allimageGPU_denominator,float *numerator_pp,float *numerator_ps,float *denominator,\
	float *single_pp_grad,float *single_ps_grad,float *pp_gradient,float *ps_gradient,float *fw_vpx,float *fw_vpz,float *fw_vsx,float *fw_vsz,float *vpx_g,float *vpz_g,float *vsx_g,float *vsz_g,\
	float *bw_vpx,float *bw_vpz,float *bw_vsx,float *bw_vsz,float *vx,float *vz,const int nx,const int nz,const int ngx_left,const int pml,const int nzpml){

	int iz = blockIdx.x*blockDim.x+threadIdx.x;
	int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(ix>nx-1||iz>nz-1)return;
//	float illumination_pp = fw_vpx[(ix+pml)*nzpml+iz+pml]*fw_vpx[(ix+pml)*nzpml+iz+pml]+fw_vpz[(ix+pml)*nzpml+iz+pml]*fw_vpz[(ix+pml)*nzpml+iz+pml];	
	float cross_correlation_pp = fw_vpx[(ix+pml)*nzpml+iz+pml]*bw_vpx[(ix+pml)*nzpml+iz+pml]+fw_vpz[(ix+pml)*nzpml+iz+pml]*bw_vpz[(ix+pml)*nzpml+iz+pml];
    float cross_correlation_ps = fw_vpx[(ix+pml)*nzpml+iz+pml]*bw_vsx[(ix+pml)*nzpml+iz+pml]+fw_vpz[(ix+pml)*nzpml+iz+pml]*bw_vsz[(ix+pml)*nzpml+iz+pml];		

//	float cross_correlation_ps = (fw_vpx[(ix+pml)*nzpml+iz+pml]+fw_vpz[(ix+pml)*nzpml+iz+pml])*(bw_vsx[(ix+pml)*nzpml+iz+pml]+bw_vsz[(ix+pml)*nzpml+iz+pml]);	
	float illumination = vx[(ix+pml)*nzpml+iz+pml]*vx[(ix+pml)*nzpml+iz+pml]+vz[(ix+pml)*nzpml+iz+pml]*vz[(ix+pml)*nzpml+iz+pml];

	numerator_pp[ix*nz+iz] += cross_correlation_pp;
	numerator_ps[ix*nz+iz] += cross_correlation_ps;		
	denominator[ix*nz+iz] += illumination;	

	allimageGPU_numerator_pp[(ngx_left+ix)*nz+iz] += numerator_pp[ix*nz+iz];
	allimageGPU_numerator_ps[(ngx_left+ix)*nz+iz] += numerator_ps[ix*nz+iz];	
	allimageGPU_denominator[(ngx_left+ix)*nz+iz] += denominator[ix*nz+iz];	

	single_pp_grad[ix*nz+iz] += 2*(vpx_g[(ix+pml)*nzpml+iz+pml]*bw_vpx[(ix+pml)*nzpml+iz+pml]+vpz_g[(ix+pml)*nzpml+iz+pml]*bw_vpz[(ix+pml)*nzpml+iz+pml]);
	single_ps_grad[ix*nz+iz] += (4*(vpx_g[(ix+pml)*nzpml+iz+pml]*bw_vpx[(ix+pml)*nzpml+iz+pml]+vpz_g[(ix+pml)*nzpml+iz+pml]*bw_vpz[(ix+pml)*nzpml+iz+pml]) - \
								  2*(vsx_g[(ix+pml)*nzpml+iz+pml]*bw_vsx[(ix+pml)*nzpml+iz+pml]+vsz_g[(ix+pml)*nzpml+iz+pml]*bw_vsz[(ix+pml)*nzpml+iz+pml]));




//	single_pp_grad[ix*nz+iz] += vpx_g[(ix+pml)*nzpml+iz+pml]*(bw_vpx[(ix+pml)*nzpml+iz+pml]+bw_vpz[(ix+pml)*nzpml+iz+pml]);
//	single_ps_grad[ix*nz+iz] += vsx_g[(ix+pml)*nzpml+iz+pml]*(bw_vsx[(ix+pml)*nzpml+iz+pml]+bw_vsz[(ix+pml)*nzpml+iz+pml]);
	pp_gradient[(ngx_left+ix)*nz+iz] += single_pp_grad[ix*nz+iz];
	ps_gradient[(ngx_left+ix)*nz+iz] += single_ps_grad[ix*nz+iz];

}





__global__ void GPUapplysource_backward_from_record(float *vz,float *vx,float* record,float *record_x,int* gc,const int it,const float ngx_left,const int pml,const int nzpml,\
	const int ntr,const float dx,const int nt,const float dt,bool stack,int igz)
{
    const int index = blockIdx.x*blockDim.x+threadIdx.x;
	if(index>ntr-1)return;
	int igx = pml + (int)((float)(gc[index]-ngx_left)/dx);
	// vz[igx*nzpml+pml] =  record[(index*nt)+it];	//  +=   or   =     NOT SURE	
	// vx[igx*nzpml+pml] =  record_x[(index*nt)+it];
	if(!stack){
		vz[igx*nzpml+igz] =  record[(index*nt)+it];	//  +=   or   =     NOT SURE	
		vx[igx*nzpml+igz] =  record_x[(index*nt)+it];
	}
	else{
		vz[igx*nzpml+igz] +=  record[(index*nt)+it];	//  +=   or   =     NOT SURE	
		vx[igx*nzpml+igz] +=  record_x[(index*nt)+it];		
	}
}

void FD2DGPU_ELASTIC::elastic_rotation_addS_bac(const int it,cudaStream_t stream,bool stack){
//    GPUapplysource_backward_from_record<<<adds_bac.grid,adds_bac.block,0,stream>>>(vz_bk,vx_bk,GPUrecord,GPUrecord_x,GPUgc,it,cmin,pml,nzpml,ntr,dx,nt);	
    GPUapplysource_backward_from_record<<<adds_bac.grid,adds_bac.block,0,stream>>>(vz_bk,vx_bk,GPUrecord_scatter_z,GPUrecord_scatter_x,GPUgc,it,ngx_left,pml,nzpml,ntr,dx,nt,dt,stack,igz);		
}

void FD2DGPU_ELASTIC::image(cudaStream_t stream){
	doimage<<<imagesize.grid,imagesize.block,0,stream>>>(single_image,single_image_ps,vpx,vpz,vpx_bk,vpz_bk,vsx_bk,vsz_bk,nx,nz,pml,nzpml);
}

void FD2DGPU_ELASTIC::image_gradient(cudaStream_t stream){
//	doimage_gradient<<<imagesize.grid,imagesize.block,0,stream>>>(allimageGPU_numerator_pp,allimageGPU_numerator_ps,allimageGPU_denominator,numerator_pp,numerator_ps,denominator_pp,vx_x,vz_z,vx_z,vz_x,txx_bk,tzz_bk,txz_bk,vp,vs,vx,vz,d_rho,nx,nz,cmin,pml,nzpml,dt);
	doimage_gradient<<<imagesize.grid,imagesize.block,0,stream>>>(pp_gradient,ps_gradient,vx_gx,vz_gz,vx_gz,vz_gx,txx_bk,tzz_bk,txz_bk,vp,vs,d_rho,nx,nz,delta_left,record_left_in_v,pml,nzpml);	// vpx_g vpz_g
	// doimage_gradient<<<imagesize.grid,imagesize.block,0,stream>>>(b_allimageGPU_numerator_pp,b_allimageGPU_numerator_ps,b_vx_gx,b_vz_gz,b_vx_gz,b_vz_gx,\
	// txx_bk,tzz_bk,txz_bk,vp,vs,d_rho,nx,nz,delta_left,record_left_in_v,pml,nzpml);			// forward scatter G1 * receiver delta_d G0
	// doimage_gradient<<<imagesize.grid,imagesize.block,0,stream>>>(allimageGPU_numerator_pp,allimageGPU_numerator_ps,vx_gx,vz_gz,vx_gz,vz_gx,\
	// b_txx_bk,b_tzz_bk,b_txz_bk,vp,vs,d_rho,nx,nz,delta_left,record_left_in_v,pml,nzpml);			// forward G0 * receiver scatter G1

	doimage_illum<<<imagesize.grid,imagesize.block,0,stream>>>(allimageGPU_denominator,vx,vz,nx,nz,delta_left,record_left_in_v,pml,nzpml);	


//	doimage_gradient<<<imagesize.grid,imagesize.block,0,stream>>>(allimageGPU_numerator_pp,allimageGPU_numerator_ps,allimageGPU_denominator,numerator_pp, \
		numerator_ps,denominator_pp,vpx_g,vpz_g,vsx_g,vsz_g,vpx_bk,vpz_bk,vsx_bk,vsz_bk,nx,nz,cmin,pml,nzpml);
//	doimage_gradient<<<imagesize.grid,imagesize.block,0,stream>>>(allimageGPU_numerator_pp,allimageGPU_numerator_ps,allimageGPU_denominator,numerator_pp, \
		numerator_ps,denominator_pp,single_pp_grad,single_ps_grad,pp_gradient,ps_gradient,vpx,vpz,vsx,vsz,vpx_g,vpz_g,vsx_g,vsz_g,vpx_bk,vpz_bk,vsx_bk,vsz_bk,vx,vz,nx,nz,cmin,pml,nzpml);

}


void FD2DGPU_ELASTIC::record_copytoGPU(float* record,float* record_x,cudaStream_t stream){	
	cudaMemcpy(GPUrecord,record,nt*ntr*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(GPUrecord_x,record_x,nt*ntr*sizeof(float),cudaMemcpyHostToDevice);	
}

void FD2DGPU_ELASTIC::record_copytoGPU(float* record,float* record_x,int* gc,cudaStream_t stream){

	static int count = 0;

	static int old_ntr = ntr;
	// std::cout<<"111 old_ntr = "<<old_ntr<<std::endl;

	if((GPUrecord!=NULL&&ntr!=old_ntr)||count == 0){
		std::cout<<"***** record is not nullptr and ntr changed OR this is count 0. *****"<<std::endl;
		cudaFree(GPUgc);
		cudaFree(GPUrecord);
		cudaFree(GPUrecord_x);
//		float *GPUgc,*GPUrecord,*GPUrecord_x;
		cudaMalloc((void**)&GPUgc,ntr*sizeof(int));
		cudaMalloc((void**)&GPUrecord,ntr*nt*sizeof(float));
		cudaMalloc((void**)&GPUrecord_x,ntr*nt*sizeof(float));
		old_ntr = ntr;
	}

	// std::cout<<"222 old_ntr = "<<old_ntr<<std::endl;

	cudaMemcpy(GPUgc,gc,ntr*sizeof(int),cudaMemcpyHostToDevice);		
	cudaMemcpy(GPUrecord,record,nt*ntr*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(GPUrecord_x,record_x,nt*ntr*sizeof(float),cudaMemcpyHostToDevice);	

	count++;
}



void FD2DGPU_ELASTIC::rot_exp_calculate_v(int direction,bool forward,cudaStream_t stream,bool rbc){
	if(forward){
		if(rbc){
			GPUcalculate_rotation_v<<<Grid_Block.grid,Grid_Block.block,0,stream>>>(theta,omega,vx,vz,vpx,vpz,vsx,vsz,vp,vs,nxpml,nzpml,dt,dx,dz,nop,d_rho,direction);		
		}
		else{
			GPUcalculate_rotation_v<<<Grid_Block.grid,Grid_Block.block,0,stream>>>(theta,omega,vx,vz,vpx,vpz,vsx,vsz,vp,vs,nxpml,nzpml,dt,dx,dz,nop,d_rho,direction,d_damp,d_dampz);	
		}
	}
	else{
		if(rbc){
			GPUcalculate_rotation_v<<<Grid_Block.grid,Grid_Block.block,0,stream>>>(theta_bk,omega_bk,vx_bk,vz_bk,vpx_bk,vpz_bk,vsx_bk,vsz_bk,vp,vs,nxpml,nzpml,dt,dx,dz,nop,d_rho,direction); //rbc
		}
		else{
			GPUcalculate_rotation_v<<<Grid_Block.grid,Grid_Block.block,0,stream>>>(theta_bk,omega_bk,vx_bk,vz_bk,vpx_bk,vpz_bk,vsx_bk,vsz_bk,vp,vs,nxpml,nzpml,dt,dx,dz,nop,d_rho,direction,d_damp,d_dampz);	
		}			
	}
		
};
void FD2DGPU_ELASTIC::rot_exp_calculate_p(int direction,bool forward,cudaStream_t stream,bool rbc){
	if(forward){
		if(rbc){	
			GPUcalculate_rotation_p<<<Grid_Block.grid,Grid_Block.block,0,stream>>>(theta,omega,vx,vz,theta_x,theta_z,omega_x,omega_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,direction);
		}
		else{
			GPUcalculate_rotation_p<<<Grid_Block.grid,Grid_Block.block,0,stream>>>(theta,omega,vx,vz,theta_x,theta_z,omega_x,omega_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,direction,d_damp,d_dampz);		
		}
	}
	else{
		if(rbc){
			GPUcalculate_rotation_p<<<Grid_Block.grid,Grid_Block.block,0,stream>>>(theta_bk,omega_bk,vx_bk,vz_bk,theta_x_bk,theta_z_bk,omega_x_bk,omega_z_bk,nxpml,nzpml,dt,dx,dz,nop,d_rho,direction); // random boundary
		}
		else{
			GPUcalculate_rotation_p<<<Grid_Block.grid,Grid_Block.block,0,stream>>>(theta_bk,omega_bk,vx_bk,vz_bk,theta_x_bk,theta_z_bk,omega_x_bk,omega_z_bk,nxpml,nzpml,dt,dx,dz,nop,d_rho,direction,d_damp,d_dampz); 		
		}
	}
};


void FD2DGPU_ELASTIC::rot_exp_calculate_v_born(int direction,bool forward,bool reconstruct,cudaStream_t stream,bool rbc){
	if(reconstruct==false){
		if(rbc){	
			GPUcalculate_rotation_v_born<<<Grid_Block.grid,Grid_Block.block,0,stream>>>(theta,omega,vx,vz,vpx,vpz,vsx,vsz,vp,vs,nxpml,nzpml,dt,dx,dz,nop,d_rho,direction,\
				b_theta,b_omega,b_vpx,b_vpz,b_vsx,b_vsz,b_vx,b_vz,s_image_pp_m,s_image_ps_m);
		}
		else{
			GPUcalculate_rotation_v_born<<<Grid_Block.grid,Grid_Block.block,0,stream>>>(theta,omega,vx,vz,vpx,vpz,vsx,vsz,vp,vs,nxpml,nzpml,dt,dx,dz,nop,d_rho,direction,\
				b_theta,b_omega,b_vpx,b_vpz,b_vsx,b_vsz,b_vx,b_vz,s_image_pp_m,s_image_ps_m,d_damp,d_dampz);		
		}
	}
	else{
		if(rbc){
			GPUcalculate_rotation_v_reconstruct<<<Grid_Block.grid,Grid_Block.block,0,stream>>>(theta,omega,vx,vz,vpx,vpz,vsx,vsz,vp,vs,nxpml,nzpml,dt,dx,dz,nop,d_rho,direction,\
				vpx_g,vpz_g,vsx_g,vsz_g);
		}
		else{
			GPUcalculate_rotation_v_reconstruct<<<Grid_Block.grid,Grid_Block.block,0,stream>>>(theta,omega,vx,vz,vpx,vpz,vsx,vsz,vp,vs,nxpml,nzpml,dt,dx,dz,nop,d_rho,direction,\
				vpx_g,vpz_g,vsx_g,vsz_g,d_damp,d_dampz);
		}				
	}
}

void FD2DGPU_ELASTIC::rot_exp_calculate_p_born(int direction,bool forward,cudaStream_t stream,bool rbc){
	if(rbc){
		GPUcalculate_rotation_p_born<<<Grid_Block.grid,Grid_Block.block,0,stream>>>(theta,omega,vx,vz,theta_x,theta_z,omega_x,omega_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,direction,\
				b_vx,b_vz,b_theta_x,b_theta_z,b_omega_x,b_omega_z,b_theta,b_omega);
	}
	else{
		GPUcalculate_rotation_p_born<<<Grid_Block.grid,Grid_Block.block,0,stream>>>(theta,omega,vx,vz,theta_x,theta_z,omega_x,omega_z,nxpml,nzpml,dt,dx,dz,nop,d_rho,direction,\
				b_vx,b_vz,b_theta_x,b_theta_z,b_omega_x,b_omega_z,b_theta,b_omega,d_damp,d_dampz);	
	}
}

void FD2DGPU_ELASTIC::record_scatter(const int it,bool double_scatter){
	dim3 gridsize((ntr+BLOCKDIMX-1)/BLOCKDIMX,1);
	dim3 blocksize(BLOCKDIMX,1);
	GPUrecord_scatterwavefield<<<gridsize,blocksize>>>(b_vz,b_vx,GPUrecord_scatter_z,GPUrecord_scatter_x,GPUgc,it,ngx_left,pml,nzpml,ntr,dx,nt,igz);
	if(double_scatter){
		// GPUrecord_scatterwavefield<<<gridsize,blocksize>>>(b2_vz,b2_vx,GPUrecord_scatter2_z,GPUrecord_scatter2_x,GPUgc,it,ngx_left,pml,nzpml,ntr,dx,nt,igz);	
		GPUrecord_scatterwavefield<<<gridsize,blocksize>>>(vz,vx,GPUrecord_scatter2_z,GPUrecord_scatter2_x,GPUgc,it,ngx_left,pml,nzpml,ntr,dx,nt,igz);				
	}
}

__global__ void calculate_image(float *allimageGPU_numerator_pp,float *allimageGPU_numerator_ps,float *allimageGPU_denominator,const int allnx,const int allnz){
	int iz = blockIdx.x*blockDim.x+threadIdx.x;
	int ix = blockIdx.y*blockDim.y+threadIdx.y;
	float epsilon = 1e-6;
	if(ix>allnx-1||iz>allnz-1)return;
	allimageGPU_numerator_pp[ix*allnz+iz] = allimageGPU_numerator_pp[ix*allnz+iz]/(allimageGPU_denominator[ix*allnz+iz]+epsilon);
	allimageGPU_numerator_ps[ix*allnz+iz] = allimageGPU_numerator_ps[ix*allnz+iz]/(allimageGPU_denominator[ix*allnz+iz]+epsilon);
}

void FD2DGPU_ELASTIC::imagebuffer_resettozero(float *cpu_pp_grad,float *cpu_ps_grad,float *illumination,float *b_cpu_pp_grad,float *b_cpu_ps_grad,float *b_cpu_pp_grad2,float *b_cpu_ps_grad2){
	dim3 gridimage((allnz+BLOCKDIMX-1)/BLOCKDIMX,(allnx+BLOCKDIMY-1)/BLOCKDIMY);
	dim3 blockimage(BLOCKDIMX,BLOCKDIMY);

	cudaMemcpy(cpu_pp_grad,pp_gradient,allnx*allnz*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(cpu_ps_grad,ps_gradient,allnx*allnz*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(illumination,allimageGPU_denominator,allnx*allnz*sizeof(float),cudaMemcpyDeviceToHost);	

//reset to zero	
	cudaMemset(pp_gradient,0,allnx*allnz*sizeof(float));
	cudaMemset(ps_gradient,0,allnx*allnz*sizeof(float));
	cudaMemset(allimageGPU_denominator,0,allnx*allnz*sizeof(float));	

//////////////////////////////
	cudaMemcpy(b_cpu_pp_grad,b_allimageGPU_numerator_pp,allnx*allnz*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(b_cpu_ps_grad,b_allimageGPU_numerator_ps,allnx*allnz*sizeof(float),cudaMemcpyDeviceToHost);
//reset to zero	
	cudaMemset(b_allimageGPU_numerator_pp,0,allnx*allnz*sizeof(float));
	cudaMemset(b_allimageGPU_numerator_ps,0,allnx*allnz*sizeof(float));


//////////////////////////////
	cudaMemcpy(b_cpu_pp_grad2,allimageGPU_numerator_pp,allnx*allnz*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(b_cpu_ps_grad2,allimageGPU_numerator_ps,allnx*allnz*sizeof(float),cudaMemcpyDeviceToHost);
//reset to zero	
	cudaMemset(allimageGPU_numerator_pp,0,allnx*allnz*sizeof(float));
	cudaMemset(allimageGPU_numerator_ps,0,allnx*allnz*sizeof(float));
}


/*
void ROT_EXP_IMAGE_SINGLESHOT(FD2DGPU_ELASTIC &fdtdgpu,const int sxnum,const int nDim,const int myid,float *cpu_single_image,float *cpu_single_image_ps,float *record,float *record_x,int *gc){


	fdtdgpu.FD_ELASTIC_initial(nDim);
	fdtdgpu.BK_ELASTIC_initial(nDim);	

//	fdtdgpu.calculate_elastic_damp_C();				// using vt present vp
	fdtdgpu.set_random_boundary();

	// float* bkp = new float[nDim];
	// memset(bkp,0,sizeof(float)*nDim);
	// char tmp1[1024];
	// cudaMemcpy(bkp,fdtdgpu.vs,nDim*sizeof(float),cudaMemcpyDeviceToHost);
	// sprintf(tmp1, "snap/vs_randomnew_%d_%d.dat",fdtdgpu.nxpml,fdtdgpu.nzpml);
	// write_1d_float_wb(bkp,nDim,tmp1);
	// char tmp2[1024];
	// cudaMemcpy(bkp,fdtdgpu.vp,nDim*sizeof(float),cudaMemcpyDeviceToHost);
	// sprintf(tmp2, "snap/vp_randomnew_%d_%d.dat",fdtdgpu.nxpml,fdtdgpu.nzpml);
	// write_1d_float_wb(bkp,nDim,tmp2);

    int it;
	int direction = 1;
	int direction_rec = -1;	
	bool forward = true;
	bool backward = false;	

//create CUDA stream
	cudaStream_t stream[2];
	for (int i = 0; i < 2; i++) {
  	cudaStreamCreate(&stream[i]);
	}
//

// Async memory copy
    fdtdgpu.record_copytoGPU(record,record_x,gc,stream[0]);
//
	for (it=0; it<fdtdgpu.nt; ++it)    
    {

        fdtdgpu.elastic_rotation_addS_for(it,stream[1]);
    	fdtdgpu.rot_exp_calculate_v(direction,forward,stream[1]);
    	fdtdgpu.rot_exp_calculate_p(direction,forward,stream[1]); 
	}
	cudaDeviceSynchronize();
//	clock
    struct timeval t1,t2;
    double timeuse;
    gettimeofday(&t1,NULL);
//
////// 	backward && do imaging/////
	for (it=fdtdgpu.nt-1; it>=0; --it)    
    {
//random boundary wave-field reconstruct 
    	fdtdgpu.rot_exp_calculate_p(direction_rec,forward,stream[0]);
    	fdtdgpu.rot_exp_calculate_v(direction_rec,forward,stream[0]); 	
        fdtdgpu.elastic_rotation_addS_for(it,stream[0]);
//	reverse time extrapolation
    	fdtdgpu.rot_exp_calculate_v(direction,backward,stream[1]);
        fdtdgpu.elastic_rotation_addS_bac(it,stream[1]);
    	fdtdgpu.rot_exp_calculate_p(direction,backward,stream[1]); 

		cudaStreamSynchronize(stream[0]);
		cudaStreamSynchronize(stream[1]);
//		cudaDeviceSynchronize();
/////	image part ////
		fdtdgpu.image(stream[0]);
	}
////// 	backward /////
	cudaStreamSynchronize(stream[0]);
	cudaStreamSynchronize(stream[1]);
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
    gettimeofday(&t2,NULL);
    timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
    cout<<"time = "<<timeuse<<endl;
// copy image from GPU to CPU
	char path_pp[1024];
	cudaMemcpy(cpu_single_image,fdtdgpu.single_image,fdtdgpu.nx*fdtdgpu.nz*sizeof(float),cudaMemcpyDeviceToHost);
	sprintf(path_pp, "snap/pp_single_image_%d_%d_%d.dat",sxnum,fdtdgpu.nx,fdtdgpu.nz);
	write_1d_float_wb(cpu_single_image,fdtdgpu.nx*fdtdgpu.nz,path_pp);

	char path_ps[1024];
	cudaMemcpy(cpu_single_image_ps,fdtdgpu.single_image_ps,fdtdgpu.nx*fdtdgpu.nz*sizeof(float),cudaMemcpyDeviceToHost);
	sprintf(path_ps, "snap/ps_single_image_%d_%d_%d.dat",sxnum,fdtdgpu.nx,fdtdgpu.nz);
	write_1d_float_wb(cpu_single_image_ps,fdtdgpu.nx*fdtdgpu.nz,path_ps);

//
}
*/
__global__ void record_edge_wavefield(float *omega,float *theta,float *txz,float *GPU_omega_up,float *GPU_theta_up,float *GPU_omega_down,float *GPU_theta_down, \
		float *GPU_omega_left,float *GPU_theta_left,float *GPU_omega_right,float *GPU_theta_right,\
		float *GPU_txz_up,float *GPU_txz_down,float *GPU_txz_left,float *GPU_txz_right,
		const int pml,const int nzpml,const int nxpml,const int nx,const int nz,const int N,const int it){

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;	

	if(ix>=pml&&ix<nxpml-pml&&iz>=pml-N&&iz<pml)
	{
		GPU_omega_up[it*nx*N+(ix-pml)*N+(iz-(pml-N))] = omega[ix*nzpml+iz];
		GPU_theta_up[it*nx*N+(ix-pml)*N+(iz-(pml-N))] = theta[ix*nzpml+iz];
		GPU_txz_up[it*nx*N+(ix-pml)*N+(iz-(pml-N))] = txz[ix*nzpml+iz];


		GPU_omega_down[it*nx*N+(ix-pml)*N+(iz-(pml-N))] = omega[ix*nzpml+(nzpml-iz)];
		GPU_theta_down[it*nx*N+(ix-pml)*N+(iz-(pml-N))] = theta[ix*nzpml+(nzpml-iz)];
		GPU_txz_down[it*nx*N+(ix-pml)*N+(iz-(pml-N))] = txz[ix*nzpml+(nzpml-iz)];
	}

	if(ix>=pml-N&&ix<pml&&iz>=pml-N&&iz<nz+pml+N)
	{

		GPU_omega_left[it*N*(nz+2*N)+(ix-(pml-N))*(nz+2*N)+(iz-(pml-N))] = omega[ix*nzpml+iz];
		GPU_theta_left[it*N*(nz+2*N)+(ix-(pml-N))*(nz+2*N)+(iz-(pml-N))] = theta[ix*nzpml+iz];
		GPU_txz_left[it*N*(nz+2*N)+(ix-(pml-N))*(nz+2*N)+(iz-(pml-N))] = txz[ix*nzpml+iz];


		GPU_omega_right[it*N*(nz+2*N)+(ix-(pml-N))*(nz+2*N)+(iz-(pml-N))] = omega[(nxpml-ix)*nzpml+iz];
		GPU_theta_right[it*N*(nz+2*N)+(ix-(pml-N))*(nz+2*N)+(iz-(pml-N))] = theta[(nxpml-ix)*nzpml+iz];
		GPU_txz_right[it*N*(nz+2*N)+(ix-(pml-N))*(nz+2*N)+(iz-(pml-N))] = txz[(nxpml-ix)*nzpml+iz];
	}

}




__global__ void load_edge_wavefield(float *omega,float *theta,float *txz,float *GPU_omega_up,float *GPU_theta_up,float *GPU_omega_down,float *GPU_theta_down, \
		float *GPU_omega_left,float *GPU_theta_left,float *GPU_omega_right,float *GPU_theta_right,\
		float *GPU_txz_up,float *GPU_txz_down,float *GPU_txz_left,float *GPU_txz_right,		
		const int pml,const int nzpml,const int nxpml,const int nx,const int nz,const int N,const int it){

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;	

	if(ix>=pml&&ix<nxpml-pml&&iz>=pml-N&&iz<pml)
	{
		omega[ix*nzpml+iz] = GPU_omega_up[it*nx*N+(ix-pml)*N+(iz-(pml-N))];
		theta[ix*nzpml+iz] = GPU_theta_up[it*nx*N+(ix-pml)*N+(iz-(pml-N))];
		txz[ix*nzpml+iz] = GPU_txz_up[it*nx*N+(ix-pml)*N+(iz-(pml-N))];

		omega[ix*nzpml+(nzpml-iz)] = GPU_omega_down[it*nx*N+(ix-pml)*N+(iz-(pml-N))];
		theta[ix*nzpml+(nzpml-iz)] = GPU_theta_down[it*nx*N+(ix-pml)*N+(iz-(pml-N))];
		txz[ix*nzpml+(nzpml-iz)] = GPU_txz_down[it*nx*N+(ix-pml)*N+(iz-(pml-N))];
	}

	if(ix>=pml-N&&ix<pml&&iz>=pml-N&&iz<nz+pml+N)
	{

		omega[ix*nzpml+iz] = GPU_omega_left[it*N*(nz+2*N)+(ix-(pml-N))*(nz+2*N)+(iz-(pml-N))];
		theta[ix*nzpml+iz] = GPU_theta_left[it*N*(nz+2*N)+(ix-(pml-N))*(nz+2*N)+(iz-(pml-N))];
		txz[ix*nzpml+iz] = GPU_txz_left[it*N*(nz+2*N)+(ix-(pml-N))*(nz+2*N)+(iz-(pml-N))];		

		omega[(nxpml-ix)*nzpml+iz] = GPU_omega_right[it*N*(nz+2*N)+(ix-(pml-N))*(nz+2*N)+(iz-(pml-N))];
		theta[(nxpml-ix)*nzpml+iz] = GPU_theta_right[it*N*(nz+2*N)+(ix-(pml-N))*(nz+2*N)+(iz-(pml-N))];
		txz[(nxpml-ix)*nzpml+iz] = GPU_txz_right[it*N*(nz+2*N)+(ix-(pml-N))*(nz+2*N)+(iz-(pml-N))];
	}

}






__global__ void record_edge_wavefield(float *omega,float *theta,float *GPU_omega_up,float *GPU_theta_up,float *GPU_omega_down,float *GPU_theta_down, \
		float *GPU_omega_left,float *GPU_theta_left,float *GPU_omega_right,float *GPU_theta_right,const int pml,const int nzpml,const int nxpml,const int nx,const int nz,const int N,const int it){

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;	

	if(ix>=pml&&ix<nxpml-pml&&iz>=pml-N&&iz<pml+N)
	{
		GPU_omega_up[it*nx*2*N+(ix-pml)*2*N+(iz-(pml-N))] = omega[ix*nzpml+iz];
		GPU_theta_up[it*nx*2*N+(ix-pml)*2*N+(iz-(pml-N))] = theta[ix*nzpml+iz];
	}
	if(ix>=pml&&ix<nxpml-pml&&iz>=nzpml-pml-N&&iz<nzpml-pml+N)
	{
		GPU_omega_down[it*nx*2*N+(ix-pml)*2*N+(iz-(nzpml-pml-N))] = omega[ix*nzpml+iz];
		GPU_theta_down[it*nx*2*N+(ix-pml)*2*N+(iz-(nzpml-pml-N))] = theta[ix*nzpml+iz];
	}
	if(ix>=pml-N&&ix<pml+N&&iz>=pml&&iz<nzpml-pml)
	{
		GPU_omega_left[it*2*N*nz+(ix-(pml-N))*nz+(iz-pml)] = omega[ix*nzpml+iz];
		GPU_theta_left[it*2*N*nz+(ix-(pml-N))*nz+(iz-pml)] = theta[ix*nzpml+iz];
	}
	if(ix>=nxpml-pml-N&&ix<nxpml-pml+N&&iz>=pml&&iz<nzpml-pml)
	{
		GPU_omega_right[it*2*N*nz+(ix-(nxpml-pml-N))*nz+(iz-pml)] = omega[ix*nzpml+iz];
		GPU_theta_right[it*2*N*nz+(ix-(nxpml-pml-N))*nz+(iz-pml)] = theta[ix*nzpml+iz];
	}

}



__global__ void load_edge_wavefield(float *omega,float *theta,float *GPU_omega_up,float *GPU_theta_up,float *GPU_omega_down,float *GPU_theta_down, \
		float *GPU_omega_left,float *GPU_theta_left,float *GPU_omega_right,float *GPU_theta_right,const int pml,const int nzpml,const int nxpml,const int nx,const int nz,const int N,const int it){

	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;	

	if(ix>=pml&&ix<nxpml-pml&&iz>=pml-N&&iz<pml+N)
	{
		omega[ix*nzpml+iz] = GPU_omega_up[it*nx*2*N+(ix-pml)*2*N+(iz-(pml-N))] ;
		theta[ix*nzpml+iz] = GPU_theta_up[it*nx*2*N+(ix-pml)*2*N+(iz-(pml-N))] ;
	}
	if(ix>=pml&&ix<nxpml-pml&&iz>=nzpml-pml-N&&iz<nzpml-pml+N)
	{
		omega[ix*nzpml+iz] = GPU_omega_down[it*nx*2*N+(ix-pml)*2*N+(iz-(nzpml-pml-N))];
		theta[ix*nzpml+iz] = GPU_theta_down[it*nx*2*N+(ix-pml)*2*N+(iz-(nzpml-pml-N))];
	}
	if(ix>=pml-N&&ix<pml+N&&iz>=pml&&iz<nzpml-pml)
	{
		omega[ix*nzpml+iz] = GPU_omega_left[it*2*N*nz+(ix-(pml-N))*nz+(iz-pml)];
		theta[ix*nzpml+iz] = GPU_theta_left[it*2*N*nz+(ix-(pml-N))*nz+(iz-pml)];
	}
	if(ix>=nxpml-pml-N&&ix<nxpml-pml+N&&iz>=pml&&iz<nzpml-pml)
	{
		omega[ix*nzpml+iz] = GPU_omega_right[it*2*N*nz+(ix-(nxpml-pml-N))*nz+(iz-pml)];
		theta[ix*nzpml+iz] = GPU_theta_right[it*2*N*nz+(ix-(nxpml-pml-N))*nz+(iz-pml)];
	}
}

__global__ void image_divide(const int nx,const int nz,float *illumination,float *image_PP,float *image_PS){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(ix>nx-1||iz>nz-1)return;
		image_PP[ix*nz+iz] /= illumination[ix*nz+iz];
		image_PS[ix*nz+iz] /= illumination[ix*nz+iz];

}


__global__ void image_set_top_zero(const int nx,const int nz,int layer,float *image_PP,float *image_PS){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
	if(iz<layer){
		image_PP[ix*nz+iz] = 0;
		image_PS[ix*nz+iz] = 0;
	}
}



__global__ void Low_pass(float dx, float dz, int Xn,int Zn,float* k_p, float kx_cut, float kz_cut, float taper_ratio,int k)
{
	int i, j;
	// float PI=3.1415926;
	i = blockIdx.x * blockDim.x + threadIdx.x;
	j = blockIdx.y * blockDim.y + threadIdx.y;
	float xc = kx_cut / (PI / dx / (Xn / 2));
	float zc = kz_cut / (PI / dz / (Zn / 2));
	float xs = (1 - taper_ratio) * xc;
	float zs = (1 - taper_ratio) * zc;
	int nxh = Xn / 2;
	int nzh = Zn / 2;
	int n = 2;
	if (i * Zn + j == 0)
	{
		printf("xc:%f\t\n", xc);
	}
	if (j >= 0 && j < Zn && i >= 0 && i < Xn)
	{
		k_p[i * Zn + j] = 0;
		if (k == 0)//低通滤波器
		{// filtering at x-direction
			if (i >= 0 && i < xs)
			{
				k_p[i * Zn + j] = 1.0;
			}
			else if (i >= xs && i < xc)
			{
				k_p[i * Zn + j] = cos(PI / 2.0 * (i - xs) / (xc - xs));		// cosin window
			}
			else if (i >= xc && i <= nxh)
			{
				k_p[i * Zn + j] = 0.0;
			}
			else if (i >= nxh && i < Xn - xc)
			{
				k_p[i * Zn + j] = 0.0;
			}
			else if (i >= Xn - xc && i < Xn - xs)
			{
				k_p[i * Zn + j] = sin(PI / 2.0 * (i - (Xn - xc)) / (xc - xs));
			}
			else if (i >= Xn - xs && i < Xn)
			{
				k_p[i * Zn + j] = 1.0;
			}

			// filtering at z-direction
			if (j >= 0 && j < zs)
			{
				k_p[i * Zn + j] *= 1.0;
			}
			else if (j >= zs && j < zc)
			{
				k_p[i * Zn + j] *= cos(PI / 2.0 * (j - zs) / (zc - zs));		// cosin window
			}
			else if (j >= zc && j <= nzh)
			{
				k_p[i * Zn + j] *= 0.0;
			}
			else if (j >= nzh && j < Zn - zc)
			{
				k_p[i * Zn + j] *= 0.0;
			}
			else if (j >= Zn - zc && j < Zn - zs)
			{
				k_p[i * Zn + j] *= sin(PI / 2.0 * (j - (Zn - zc)) / (zc - zs));
			}
			else if (j >= Zn - zs && j < Zn)
			{
				k_p[i * Zn + j] *= 1.0;
			}
		}
		if (k == 1)//高斯滤波器
		{
			// filtering at x-direction
			if (i >= 0 && i < nxh)
			{
				k_p[i * Zn + j] = expf(-i * i / (2 * powf(xc, 2.0)));
			}
			else
			{
				k_p[i * Zn + j] = expf(-(Xn - i) * (Xn - i) / (2 * powf(xc, 2.0)));
			}

			// filtering at z-direction
			if (j >= 0 && j < nzh)
			{
				k_p[i * Zn + j] *= expf(-j * j / (2 * powf(zc, 2.0)));
			}
			else
			{
				k_p[i * Zn + j] *= expf(-(Zn - j) * (Zn - j) / (2 * powf(zc, 2.0)));
			}
		}
		if (k == 2)//巴特沃斯滤波器
		{
			// filtering at x-direction
			if (i >= 0 && i < nxh)
			{
				k_p[i * Zn + j] = 1.0 / (1 + powf(i / xc, 2 * n));
			}
			else
			{
				k_p[i * Zn + j] = 1.0 / (1 + powf((Xn - i) / xc, 2 * n));
			}

			// filtering at z-direction
			if (j >= 0 && j < nzh)
			{
				k_p[i * Zn + j] *= 1.0 / (1 + powf(j / zc, 2 * n));
			}
			else
			{
				k_p[i * Zn + j] *= 1.0 / (1 + powf((Zn - j) / zc, 2 * n));
			}
		}
		// k_s[i * Zn + j] = k_p[i * Zn + j];
	}
}


__device__ cufftDoubleComplex operator/(const cufftDoubleComplex& a, const float b) {
    cufftDoubleComplex result;
    result.x = a.x/b;  // 实部
    result.y = a.y/b;  // 虚部
    return result;
}
__device__ cufftDoubleComplex operator*(const cufftDoubleComplex& a, const cufftDoubleComplex& b) {
    cufftDoubleComplex result;
    result.x = a.x * b.x - a.y * b.y;  // 实部
    result.y = a.x * b.y + a.y * b.x;  // 虚部
    return result;
}
template<typename T>
__global__ void GPU_divide(T *d_px,T *d_pz,const int nx,const int nz){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
    if(ix>nx-1||iz>nz-1)return;
    d_px[iz+ix*nz].x = d_px[iz+ix*nz].x/(nx*nz);   
    d_px[iz+ix*nz].y = d_px[iz+ix*nz].y/(nx*nz);     
    d_pz[iz+ix*nz].x = d_pz[iz+ix*nz].x/(nx*nz);   
    d_pz[iz+ix*nz].y = d_pz[iz+ix*nz].y/(nx*nz);     	      
}
template<typename T>
__global__ void GPU_lowpass(T *d_px,T *d_pz,float *k_filter,const int nx,const int nz){
	const int iz = blockIdx.x*blockDim.x+threadIdx.x;
	const int ix = blockIdx.y*blockDim.y+threadIdx.y;
    if(ix>nx-1||iz>nz-1)return;
    T filter = make_cuDoubleComplex(k_filter[ix*nz+iz],k_filter[ix*nz+iz]);
    d_px[iz+ix*nz]= d_px[iz+ix*nz]*filter;  
    d_pz[iz+ix*nz]= d_pz[iz+ix*nz]*filter;  	 
}
__global__ void R2Z(float* FloatArrayx,float* FloatArrayz,cufftDoubleComplex* complexDoubleArrayx,cufftDoubleComplex* complexDoubleArrayz,const int index){
    const int i =blockIdx.x*blockDim.x+threadIdx.x;
    if(i>index-1)return;
    complexDoubleArrayx[i].x = static_cast<double>(FloatArrayx[i]);
    complexDoubleArrayx[i].y = 0;
    complexDoubleArrayz[i].x = static_cast<double>(FloatArrayz[i]);
    complexDoubleArrayz[i].y = 0;	
}

__global__ void Z2R(float* FloatArrayx,float* FloatArrayz,cufftDoubleComplex* complexDoubleArrayx,cufftDoubleComplex* complexDoubleArrayz,const int index){
    const int i =blockIdx.x*blockDim.x+threadIdx.x;
    if(i>index-1)return;        
    FloatArrayx[i] = static_cast<float>(complexDoubleArrayx[i].x);
    FloatArrayz[i] = static_cast<float>(complexDoubleArrayz[i].x);	
}




void IMAGE_SINGLESHOT(FD2DGPU_ELASTIC &fdtdgpu,const int sxnum,const int nDim,const int myid,float *record,float *record_x,int *gc,bool rbc,bool cpu_mem,int ntt,int iter, char* snapdir,\
	float *image_pp,float *image_ps,float *cpu_pp_grad_old,float *cpu_ps_grad_old,double &misfit){


	fdtdgpu.FD_ELASTIC_initial(nDim);
	fdtdgpu.BK_ELASTIC_initial(nDim);	
	fdtdgpu.LSRTM_initial(nDim);



// 	char refl_p_path[1024];
// 	FILE *freflp = NULL;

// 	float *refl_p = new float[fdtdgpu.allnx * fdtdgpu.allnz];

// 	sprintf(refl_p_path, "/home/rondo/ertm-cuda-del/velmodel/prismlayer/refl_p_401x201.bin");
// 	freflp = fopen(refl_p_path,"rb");
// 	fread(refl_p,fdtdgpu.allnx*fdtdgpu.allnz*sizeof(float),1,freflp);
// 	fclose(freflp);	
	
// 	cudaMemcpy(fdtdgpu.image_pp_m,refl_p,fdtdgpu.allnx*fdtdgpu.allnz*sizeof(float),cudaMemcpyHostToDevice);


// // refl_s
// 	sprintf(refl_p_path, "/home/rondo/ertm-cuda-del/velmodel/prismlayer/refl_s_401x201.bin");
// 	freflp = fopen(refl_p_path,"rb");
// 	fread(refl_p,fdtdgpu.allnx*fdtdgpu.allnz*sizeof(float),1,freflp);
// 	fclose(freflp);	

// 	cudaMemcpy(fdtdgpu.image_ps_m,refl_p,fdtdgpu.allnx*fdtdgpu.allnz*sizeof(float),cudaMemcpyHostToDevice);




	fdtdgpu.buffer_image_extrapolation(fdtdgpu.cmin);
//	bool rbc = false;

	if(rbc){
		fdtdgpu.set_random_boundary();
	}
	else{
		fdtdgpu.calculate_elastic_damp_C();	
	}


	int direction = 1;
	int direction_rec = -1;					//TODO  1 OR -1
	bool forward = true;
	bool backward = false;	

	 float* bkp = new float[nDim];
	 memset(bkp,0,sizeof(float)*nDim);

	bool reconst = false;

// Async memory copy
    // fdtdgpu.record_copytoGPU(record,record_x,gc,stream[0]);
//

// storage edge wavefield of theta & omega

		int nx = fdtdgpu.nx;
		int nt = fdtdgpu.nt;
		int pml = fdtdgpu.pml;
		int nzpml = fdtdgpu.nzpml;
		int nz = fdtdgpu.nz;
		float *omega = fdtdgpu.omega;
		float *theta = fdtdgpu.theta;
		int N = fdtdgpu.nop;
		int nxpml = fdtdgpu.nxpml;

		float *GPU_theta_up;
		float *GPU_theta_down;
		float *GPU_omega_up;
		float *GPU_omega_down;
		float *GPU_theta_left;
		float *GPU_theta_right;
		float *GPU_omega_left;
		float *GPU_omega_right;

		float *CPU_theta_up;
		float *CPU_theta_down;
		float *CPU_omega_up;
		float *CPU_omega_down;
		float *CPU_theta_left;
		float *CPU_theta_right;
		float *CPU_omega_left;
		float *CPU_omega_right;

	if(!rbc){
		if(!cpu_mem){
			cudaMalloc((void**)&GPU_theta_up,2*N*nx*nt*sizeof(float));
			cudaMalloc((void**)&GPU_theta_down,2*N*nx*nt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_up,2*N*nx*nt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_down,2*N*nx*nt*sizeof(float));
			cudaMalloc((void**)&GPU_theta_left,2*(nz)*N*nt*sizeof(float));
			cudaMalloc((void**)&GPU_theta_right,2*(nz)*N*nt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_left,2*(nz)*N*nt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_right,2*(nz)*N*nt*sizeof(float));
		}
		else{
			CPU_theta_up = new float[2*N*nx*nt];
			CPU_theta_down = new float[2*N*nx*nt];
			CPU_omega_up = new float[2*N*nx*nt];
			CPU_omega_down = new float[2*N*nx*nt];
			CPU_theta_left = new float[(nz)*2*N*nt];
			CPU_theta_right = new float[(nz)*2*N*nt];
			CPU_omega_left = new float[(nz)*2*N*nt];
			CPU_omega_right = new float[(nz)*2*N*nt];

			memset(CPU_theta_up,0,sizeof(float)*2*N*nx*nt);
			memset(CPU_theta_down,0,sizeof(float)*2*N*nx*nt);
			memset(CPU_omega_up,0,sizeof(float)*2*N*nx*nt);
			memset(CPU_omega_down,0,sizeof(float)*2*N*nx*nt);

			memset(CPU_theta_left,0,sizeof(float)*(nz)*2*N*nt);
			memset(CPU_theta_right,0,sizeof(float)*(nz)*2*N*nt);
			memset(CPU_omega_left,0,sizeof(float)*(nz)*2*N*nt);
			memset(CPU_omega_right,0,sizeof(float)*(nz)*2*N*nt);

			cudaMalloc((void**)&GPU_theta_up,2*N*nx*ntt*sizeof(float));
			cudaMalloc((void**)&GPU_theta_down,2*N*nx*ntt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_up,2*N*nx*ntt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_down,2*N*nx*ntt*sizeof(float));
			cudaMalloc((void**)&GPU_theta_left,2*(nz)*N*ntt*sizeof(float));
			cudaMalloc((void**)&GPU_theta_right,2*(nz)*N*ntt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_left,2*(nz)*N*ntt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_right,2*(nz)*N*ntt*sizeof(float));	

		}
	}	


/////////test for reconstruct scatter wavefield

		float *GPU_theta_up2;
		float *GPU_theta_down2;
		float *GPU_omega_up2;
		float *GPU_omega_down2;
		float *GPU_theta_left2;
		float *GPU_theta_right2;
		float *GPU_omega_left2;
		float *GPU_omega_right2;

		float *CPU_theta_up2;
		float *CPU_theta_down2;
		float *CPU_omega_up2;
		float *CPU_omega_down2;
		float *CPU_theta_left2;
		float *CPU_theta_right2;
		float *CPU_omega_left2;
		float *CPU_omega_right2;

	if(!rbc){
		if(!cpu_mem){
			cudaMalloc((void**)&GPU_theta_up2,2*N*nx*nt*sizeof(float));
			cudaMalloc((void**)&GPU_theta_down2,2*N*nx*nt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_up2,2*N*nx*nt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_down2,2*N*nx*nt*sizeof(float));
			cudaMalloc((void**)&GPU_theta_left2,2*(nz)*N*nt*sizeof(float));
			cudaMalloc((void**)&GPU_theta_right2,2*(nz)*N*nt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_left2,2*(nz)*N*nt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_right2,2*(nz)*N*nt*sizeof(float));
		}
		else{
			CPU_theta_up2 = new float[2*N*nx*nt];
			CPU_theta_down2 = new float[2*N*nx*nt];
			CPU_omega_up2 = new float[2*N*nx*nt];
			CPU_omega_down2 = new float[2*N*nx*nt];
			CPU_theta_left2 = new float[(nz)*2*N*nt];
			CPU_theta_right2 = new float[(nz)*2*N*nt];
			CPU_omega_left2 = new float[(nz)*2*N*nt];
			CPU_omega_right2 = new float[(nz)*2*N*nt];

			memset(CPU_theta_up2,0,sizeof(float)*2*N*nx*nt);
			memset(CPU_theta_down2,0,sizeof(float)*2*N*nx*nt);
			memset(CPU_omega_up2,0,sizeof(float)*2*N*nx*nt);
			memset(CPU_omega_down2,0,sizeof(float)*2*N*nx*nt);

			memset(CPU_theta_left2,0,sizeof(float)*(nz)*2*N*nt);
			memset(CPU_theta_right2,0,sizeof(float)*(nz)*2*N*nt);
			memset(CPU_omega_left2,0,sizeof(float)*(nz)*2*N*nt);
			memset(CPU_omega_right2,0,sizeof(float)*(nz)*2*N*nt);

			cudaMalloc((void**)&GPU_theta_up2,2*N*nx*ntt*sizeof(float));
			cudaMalloc((void**)&GPU_theta_down2,2*N*nx*ntt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_up2,2*N*nx*ntt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_down2,2*N*nx*ntt*sizeof(float));
			cudaMalloc((void**)&GPU_theta_left2,2*(nz)*N*ntt*sizeof(float));
			cudaMalloc((void**)&GPU_theta_right2,2*(nz)*N*ntt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_left2,2*(nz)*N*ntt*sizeof(float));
			cudaMalloc((void**)&GPU_omega_right2,2*(nz)*N*ntt*sizeof(float));	

		}
	}	


/////////////////////// low-pass filter ///////////////////////////////////////
#if 0
	float max_kx = PI/fdtdgpu.dx;
	float max_kz = PI/fdtdgpu.dz;    
	float kx_cutoff = 0.98*max_kx;
	float kz_cutoff = 0.98*max_kz;
	float *k_filter;
	cudaMalloc((void**)&k_filter,nxpml*nzpml*sizeof(float));
	float ratio = 0.3;
	int filter_sort = 0;
	dim3 grid2((nxpml+32-1)/32,(nzpml+32-1)/32);
	dim3 block2(32,32);
	dim3 grid3((nxpml*nzpml+1024-1)/1024);	
	dim3 block3(1024);	
	Low_pass<<<grid2,block2>>>(fdtdgpu.dx,fdtdgpu.dz,nxpml,nzpml,k_filter,kx_cutoff,kz_cutoff,ratio,filter_sort);
	float *cout_filter = new float[nxpml*nzpml];
	cudaMemcpy(cout_filter,k_filter,nxpml*nzpml*sizeof(float),cudaMemcpyDeviceToHost);
	cufftHandle plan2D_forward;
	cufftHandle plan2D_inverse;
	cufftPlan2d(&plan2D_forward,nxpml,nzpml,CUFFT_Z2Z);
	cufftPlan2d(&plan2D_inverse,nxpml,nzpml,CUFFT_Z2Z);	
	cufftDoubleComplex *fft_vx;
	cufftDoubleComplex *fft_vz;	
	cudaMalloc((void**)&fft_vx,nxpml*nzpml*sizeof(cufftDoubleComplex));	   
	cudaMalloc((void**)&fft_vz,nxpml*nzpml*sizeof(cufftDoubleComplex));	
#endif
/////////////////////// low-pass filter ///////////////////////////////////////



//////////////////////////////////////
	bool born = true;
	
	// cudaMemcpy(fdtdgpu.GPUgc,gc,fdtdgpu.ntr*sizeof(int),cudaMemcpyHostToDevice);

	int forward_born=1;

	if(iter==0){
		forward_born=0;
	}
	bool doublescatter = true;


	// float* bkp2 = new float[nx*nz];
	// memset(bkp2,0,sizeof(float)*nx*nz);	


//	Lm 
	for(int it=0; it<fdtdgpu.nt; ++it)    
    	{
        	fdtdgpu.elastic_rotation_addS_for(it,0);
//			fdtdgpu.normal_forward_and_born(direction);

			fdtdgpu.born_modeling(direction,1,it);
// 			fdtdgpu.born_modeling(direction,0,it);
// ////////////////////////////////////////////////////////////			
// 			if(iter>=-1){
// 				fdtdgpu.born_modeling(direction,3,it);			
// 				fdtdgpu.born_modeling(direction,4,it);
// 			}
// 			fdtdgpu.born_modeling(direction,5,it);			// 2nd born approximate
///////////////////////////////////////////////////////////			
			fdtdgpu.record_scatter(it,doublescatter);
		if(!rbc){
			if(!cpu_mem){
			record_edge_wavefield<<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.vx,fdtdgpu.vz,GPU_omega_up,GPU_theta_up,GPU_omega_down,\
					GPU_theta_down,GPU_omega_left,GPU_theta_left,GPU_omega_right,GPU_theta_right,pml,nzpml,nxpml,nx,nz,N,it);
			}
			else{
				record_edge_wavefield<<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.vx,fdtdgpu.vz,GPU_omega_up,GPU_theta_up,GPU_omega_down,\
					GPU_theta_down,GPU_omega_left,GPU_theta_left,GPU_omega_right,GPU_theta_right,pml,nzpml,nxpml,nx,nz,N,it%ntt);
				if((it+1)%ntt==0){
//					std::cout<<it<<std::endl;
					cudaMemcpy(&CPU_omega_up[2*N*nx*((it+1)/ntt-1)*ntt],GPU_omega_up,2*N*nx*ntt*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&CPU_omega_down[2*N*nx*((it+1)/ntt-1)*ntt],GPU_omega_down,2*N*nx*ntt*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&CPU_theta_up[2*N*nx*((it+1)/ntt-1)*ntt],GPU_theta_up,2*N*nx*ntt*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&CPU_theta_down[2*N*nx*((it+1)/ntt-1)*ntt],GPU_theta_down,2*N*nx*ntt*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&CPU_omega_left[(nz)*2*N*((it+1)/ntt-1)*ntt],GPU_omega_left,(nz)*2*N*ntt*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&CPU_omega_right[(nz)*2*N*((it+1)/ntt-1)*ntt],GPU_omega_right,(nz)*2*N*ntt*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&CPU_theta_left[(nz)*2*N*((it+1)/ntt-1)*ntt],GPU_theta_left,(nz)*2*N*ntt*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&CPU_theta_right[(nz)*2*N*((it+1)/ntt-1)*ntt],GPU_theta_right,(nz)*2*N*ntt*sizeof(float),cudaMemcpyDeviceToHost);
				}
			}

			if(!cpu_mem){
			record_edge_wavefield<<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.b_vx,fdtdgpu.b_vz,GPU_omega_up2,GPU_theta_up2,GPU_omega_down2,\
					GPU_theta_down2,GPU_omega_left2,GPU_theta_left2,GPU_omega_right2,GPU_theta_right2,pml,nzpml,nxpml,nx,nz,N,it);
			}
			else{
				record_edge_wavefield<<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.b_vx,fdtdgpu.b_vz,GPU_omega_up2,GPU_theta_up2,GPU_omega_down2,\
					GPU_theta_down2,GPU_omega_left2,GPU_theta_left2,GPU_omega_right2,GPU_theta_right2,pml,nzpml,nxpml,nx,nz,N,it%ntt);
				if((it+1)%ntt==0){
//					std::cout<<it<<std::endl;
					cudaMemcpy(&CPU_omega_up2[2*N*nx*((it+1)/ntt-1)*ntt],GPU_omega_up2,2*N*nx*ntt*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&CPU_omega_down2[2*N*nx*((it+1)/ntt-1)*ntt],GPU_omega_down2,2*N*nx*ntt*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&CPU_theta_up2[2*N*nx*((it+1)/ntt-1)*ntt],GPU_theta_up2,2*N*nx*ntt*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&CPU_theta_down2[2*N*nx*((it+1)/ntt-1)*ntt],GPU_theta_down2,2*N*nx*ntt*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&CPU_omega_left2[(nz)*2*N*((it+1)/ntt-1)*ntt],GPU_omega_left2,(nz)*2*N*ntt*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&CPU_omega_right2[(nz)*2*N*((it+1)/ntt-1)*ntt],GPU_omega_right2,(nz)*2*N*ntt*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&CPU_theta_left2[(nz)*2*N*((it+1)/ntt-1)*ntt],GPU_theta_left2,(nz)*2*N*ntt*sizeof(float),cudaMemcpyDeviceToHost);
					cudaMemcpy(&CPU_theta_right2[(nz)*2*N*((it+1)/ntt-1)*ntt],GPU_theta_right2,(nz)*2*N*ntt*sizeof(float),cudaMemcpyDeviceToHost);
				}
			}
		}    

		//  if(it%200==0&&sxnum==1){
		//  	char tmps_bk[1024];
		//  	cudaMemcpy(bkp,fdtdgpu.vz,nDim*sizeof(float),cudaMemcpyDeviceToHost);

		// 	for(int i=0;i<nx;i++)
		// 	for(int j=0;j<nz;j++)
		// 	{
		// 		bkp2[i*nz+j] = bkp[(i+pml)*fdtdgpu.nzpml + (j+pml)];
		// 	}

		//  	sprintf(tmps_bk, "%s/snap_FW_vz_it%d_%d_%d.dat",snapdir,it,fdtdgpu.nx,fdtdgpu.nz);
		//  	write_1d_float_wb(bkp2,nx*nz,tmps_bk);
		//  }	
		//  if(it%200==0&&sxnum==1){
		//  	char tmps_bk[1024];
		//  	cudaMemcpy(bkp,fdtdgpu.b_vz,nDim*sizeof(float),cudaMemcpyDeviceToHost);

		// 	 for(int i=0;i<nx;i++)
		// 	 for(int j=0;j<nz;j++)
		// 	 {
		// 		 bkp2[i*nz+j] = bkp[(i+pml)*fdtdgpu.nzpml + (j+pml)];
		// 	 }

		//  	sprintf(tmps_bk, "%s/snap_FW_bvz_it%d_%d_%d.dat",snapdir,it,fdtdgpu.nx,fdtdgpu.nz);
		//  	write_1d_float_wb(bkp2,nx*nz,tmps_bk);
		//  }	

		//  if(it%500==0&&sxnum==23){
		//  	char tmps_bk[1024];
		//  	cudaMemcpy(bkp,fdtdgpu.b2_vz,nDim*sizeof(float),cudaMemcpyDeviceToHost);
		//  	sprintf(tmps_bk, "%s/snap_FW_b2vz_it%d_%d_%d.dat",snapdir,it,fdtdgpu.nxpml,fdtdgpu.nzpml);
		//  	write_1d_float_wb(bkp,nDim,tmps_bk);
		//  }	

		}

	cudaDeviceSynchronize();

    // fdtdgpu.record_copytoGPU(record,record_x,0);
	float *record_scatter_z = new float[fdtdgpu.ntr*fdtdgpu.nt];
	float *record_scatter_x = new float[fdtdgpu.ntr*fdtdgpu.nt];
	float *record_residual_z = new float[fdtdgpu.ntr*fdtdgpu.nt];
	float *record_residual_x = new float[fdtdgpu.ntr*fdtdgpu.nt];			
	float scale_size = 1.0;

// output Lm

	// char imagepath_pp[1024];
	// FILE *fpp = NULL;

	// cudaMemcpy(record_scatter_x,fdtdgpu.GPUrecord_scatter_x,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),cudaMemcpyDeviceToHost);
	// cudaMemcpy(record_scatter_z,fdtdgpu.GPUrecord_scatter_z,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),cudaMemcpyDeviceToHost);

	// if(sxnum==1){
	// 	sprintf(imagepath_pp, "%s/record_scatter_x_%d_%d_%d_%d.dat",snapdir,iter,sxnum,fdtdgpu.ntr,fdtdgpu.nt);
	// 	fpp = fopen(imagepath_pp,"wb");
	// 	fwrite(record_scatter_x,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),1,fpp);
	// 	fclose(fpp);
	// 	sprintf(imagepath_pp, "%s/record_scatter_z_%d_%d_%d_%d.dat",snapdir,iter,sxnum,fdtdgpu.ntr,fdtdgpu.nt);
	// 	fpp = fopen(imagepath_pp,"wb");
	// 	fwrite(record_scatter_z,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),1,fpp);
	// 	fclose(fpp);

	// }

	bool scatter2 = true;
	fdtdgpu.subtract(scale_size,scatter2);					// r  =  r - alpha*q
	cudaMemcpy(record_residual_z,fdtdgpu.GPUrecord_scatter_z,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(record_residual_x,fdtdgpu.GPUrecord_scatter_x,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),cudaMemcpyDeviceToHost);	

// //calculate misfit 
	
		for(int i=0;i<fdtdgpu.ntr;i++)
		for(int j=0;j<fdtdgpu.nt;j++){
			misfit += (record_residual_z[i*fdtdgpu.nt+j]*record_residual_z[i*fdtdgpu.nt+j] + record_residual_x[i*fdtdgpu.nt+j]*record_residual_x[i*fdtdgpu.nt+j]);						
		}	


	//output record
	// if(sxnum==19){
	// 	sprintf(imagepath_pp, "%s/record_residual_x_%d_%d_%d_%d.dat",snapdir,iter,sxnum,fdtdgpu.ntr,fdtdgpu.nt);
	// 	fpp = fopen(imagepath_pp,"wb");
	// 	fwrite(record_residual_x,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),1,fpp);
	// 	fclose(fpp);
	// 	sprintf(imagepath_pp, "%s/record_residual_z_%d_%d_%d_%d.dat",snapdir,iter,sxnum,fdtdgpu.ntr,fdtdgpu.nt);
	// 	fpp = fopen(imagepath_pp,"wb");
	// 	fwrite(record_residual_z,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),1,fpp);
	// 	fclose(fpp);
	// }


	reconst = true;
// g = LT *(Lm - d)
	bool stack = true;
	if(iter==0){
		stack = false;
	}
	stack = false;	// TODO:  if first iteration need to stack .

	for(int it=fdtdgpu.nt-1;it>=0;--it)    
        {
		if(!rbc){
			if(!cpu_mem){			
			load_edge_wavefield<<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.vx,fdtdgpu.vz,GPU_omega_up,GPU_theta_up,GPU_omega_down,\
					GPU_theta_down,GPU_omega_left,GPU_theta_left,GPU_omega_right,GPU_theta_right,pml,nzpml,nxpml,nx,nz,N,it);
			}
			else{
				if((it+1)%ntt==0&&it!=0){
//					std::cout<<it<<std::endl;
					cudaMemcpy(GPU_omega_up,&CPU_omega_up[2*N*nx*((it+1)/ntt-1)*ntt],2*N*nx*ntt*sizeof(float),cudaMemcpyHostToDevice);
					cudaMemcpy(GPU_omega_down,&CPU_omega_down[2*N*nx*((it+1)/ntt-1)*ntt],2*N*nx*ntt*sizeof(float),cudaMemcpyHostToDevice);
					cudaMemcpy(GPU_theta_up,&CPU_theta_up[2*N*nx*((it+1)/ntt-1)*ntt],2*N*nx*ntt*sizeof(float),cudaMemcpyHostToDevice);
					cudaMemcpy(GPU_theta_down,&CPU_theta_down[2*N*nx*((it+1)/ntt-1)*ntt],2*N*nx*ntt*sizeof(float),cudaMemcpyHostToDevice);
					cudaMemcpy(GPU_omega_left,&CPU_omega_left[(nz)*2*N*((it+1)/ntt-1)*ntt],(nz)*2*N*ntt*sizeof(float),cudaMemcpyHostToDevice);
					cudaMemcpy(GPU_omega_right,&CPU_omega_right[(nz)*2*N*((it+1)/ntt-1)*ntt],(nz)*2*N*ntt*sizeof(float),cudaMemcpyHostToDevice);
					cudaMemcpy(GPU_theta_left,&CPU_theta_left[(nz)*2*N*((it+1)/ntt-1)*ntt],(nz)*2*N*ntt*sizeof(float),cudaMemcpyHostToDevice);
					cudaMemcpy(GPU_theta_right,&CPU_theta_right[(nz)*2*N*((it+1)/ntt-1)*ntt],(nz)*2*N*ntt*sizeof(float),cudaMemcpyHostToDevice);
				}
				if(it<fdtdgpu.nt-1){
					load_edge_wavefield<<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.vx,fdtdgpu.vz,GPU_omega_up,GPU_theta_up,GPU_omega_down,\
						GPU_theta_down,GPU_omega_left,GPU_theta_left,GPU_omega_right,GPU_theta_right,pml,nzpml,nxpml,nx,nz,N,it%ntt);				
				}
			}

			if(!cpu_mem){			
			load_edge_wavefield<<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.b_vx,fdtdgpu.b_vz,GPU_omega_up2,GPU_theta_up2,GPU_omega_down2,\
					GPU_theta_down2,GPU_omega_left2,GPU_theta_left2,GPU_omega_right2,GPU_theta_right2,pml,nzpml,nxpml,nx,nz,N,it);
			}
			else{
				if((it+1)%ntt==0&&it!=0){
//					std::cout<<it<<std::endl;
					cudaMemcpy(GPU_omega_up2,&CPU_omega_up2[2*N*nx*((it+1)/ntt-1)*ntt],2*N*nx*ntt*sizeof(float),cudaMemcpyHostToDevice);
					cudaMemcpy(GPU_omega_down2,&CPU_omega_down2[2*N*nx*((it+1)/ntt-1)*ntt],2*N*nx*ntt*sizeof(float),cudaMemcpyHostToDevice);
					cudaMemcpy(GPU_theta_up2,&CPU_theta_up2[2*N*nx*((it+1)/ntt-1)*ntt],2*N*nx*ntt*sizeof(float),cudaMemcpyHostToDevice);
					cudaMemcpy(GPU_theta_down2,&CPU_theta_down2[2*N*nx*((it+1)/ntt-1)*ntt],2*N*nx*ntt*sizeof(float),cudaMemcpyHostToDevice);
					cudaMemcpy(GPU_omega_left2,&CPU_omega_left2[(nz)*2*N*((it+1)/ntt-1)*ntt],(nz)*2*N*ntt*sizeof(float),cudaMemcpyHostToDevice);
					cudaMemcpy(GPU_omega_right2,&CPU_omega_right2[(nz)*2*N*((it+1)/ntt-1)*ntt],(nz)*2*N*ntt*sizeof(float),cudaMemcpyHostToDevice);
					cudaMemcpy(GPU_theta_left2,&CPU_theta_left2[(nz)*2*N*((it+1)/ntt-1)*ntt],(nz)*2*N*ntt*sizeof(float),cudaMemcpyHostToDevice);
					cudaMemcpy(GPU_theta_right2,&CPU_theta_right2[(nz)*2*N*((it+1)/ntt-1)*ntt],(nz)*2*N*ntt*sizeof(float),cudaMemcpyHostToDevice);
				}
				if(it<fdtdgpu.nt-1){
					load_edge_wavefield<<<fdtdgpu.Grid_Block.grid,fdtdgpu.Grid_Block.block>>>(fdtdgpu.b_vx,fdtdgpu.b_vz,GPU_omega_up2,GPU_theta_up2,GPU_omega_down2,\
						GPU_theta_down2,GPU_omega_left2,GPU_theta_left2,GPU_omega_right2,GPU_theta_right2,pml,nzpml,nxpml,nx,nz,N,it%ntt);				
				}
			}

		}				

//wavefield reconstruction			
			// fdtdgpu.calculateTxz(direction_rec,forward);
			// fdtdgpu.calculateTxxzz(direction_rec,forward);
			// fdtdgpu.calculateVz(direction_rec,forward); 				
			// fdtdgpu.calculateVx(direction_rec,forward);

			fdtdgpu.born_modeling(direction_rec,6,it);



//	        	fdtdgpu.elastic_rotation_addS_for(it,0);

			// fdtdgpu.pure_born_reconstruct_backward(direction_rec,forward);			//reconstruct source-side scatter wavefield

			// if(it%1000==0&&sxnum==23){
			// 	char tmps_bk[1024];
			// 	cudaMemcpy(bkp,fdtdgpu.vz,nDim*sizeof(float),cudaMemcpyDeviceToHost);
			// 	sprintf(tmps_bk, "%s/snap_recFW_vz_it%d_%d_%d.dat",snapdir,it,fdtdgpu.nxpml,fdtdgpu.nzpml);
			// 	write_1d_float_wb(bkp,nDim,tmps_bk);
			// }
			// if(it%500==0&&sxnum==23){
			// 	char tmps_bk[1024];
			// 	cudaMemcpy(bkp,fdtdgpu.b_vz,nDim*sizeof(float),cudaMemcpyDeviceToHost);
			// 	sprintf(tmps_bk, "%s/snap_recFW_bvz_it%d_%d_%d.dat",snapdir,it,fdtdgpu.nxpml,fdtdgpu.nzpml);
			// 	write_1d_float_wb(bkp,nDim,tmps_bk);
			// }

// backward wavefield
        	fdtdgpu.elastic_rotation_addS_bac(it,0,stack);
			// fdtdgpu.calculateTxz(direction,backward);
			// fdtdgpu.calculateTxxzz(direction,backward);
			// fdtdgpu.calculateVz(direction,backward); 				
			// fdtdgpu.calculateVx(direction,backward);

			fdtdgpu.born_modeling(direction_rec,8,it);



			// fdtdgpu.pure_born_reconstruct_backward(direction,backward);			//receiver-side scatter wavefield

			// if(it%1000==0&&sxnum==23){
			// 	char tmps_bk[1024];
			// 	cudaMemcpy(bkp,fdtdgpu.vz_bk,nDim*sizeof(float),cudaMemcpyDeviceToHost);
			// 	sprintf(tmps_bk, "%s/snap_BW_vz_it%d_%d_%d.dat",snapdir,it,fdtdgpu.nxpml,fdtdgpu.nzpml);
			// 	write_1d_float_wb(bkp,nDim,tmps_bk);
			// }
			// if(it%1000==0&&sxnum==23){
			// 	char tmps_bk[1024];
			// 	cudaMemcpy(bkp,fdtdgpu.b_vz_bk,nDim*sizeof(float),cudaMemcpyDeviceToHost);
			// 	sprintf(tmps_bk, "%s/snap_BW_bvz_it%d_%d_%d.dat",snapdir,it,fdtdgpu.nxpml,fdtdgpu.nzpml);
			// 	write_1d_float_wb(bkp,nDim,tmps_bk);
			// }

			// if(it%1000==0&&sxnum==23){
			// 	char tmps_bk[1024];
			// 	cudaMemcpy(bkp,fdtdgpu.tzz_bk,nDim*sizeof(float),cudaMemcpyDeviceToHost);
			// 	sprintf(tmps_bk, "%s/snap_BW_tzz_it%d_%d_%d.dat",snapdir,it,fdtdgpu.nxpml,fdtdgpu.nzpml);
			// 	write_1d_float_wb(bkp,nDim,tmps_bk);
			// }

			// if(it%1000==0&&sxnum==23){
			// 	char tmps_bk[1024];
			// 	cudaMemcpy(bkp,fdtdgpu.b_tzz_bk,nDim*sizeof(float),cudaMemcpyDeviceToHost);
			// 	sprintf(tmps_bk, "%s/snap_BW_btzz_it%d_%d_%d.dat",snapdir,it,fdtdgpu.nxpml,fdtdgpu.nzpml);
			// 	write_1d_float_wb(bkp,nDim,tmps_bk);
			// }
	 		cudaDeviceSynchronize();			
			fdtdgpu.image_gradient(0);

		//  if(it%1000==0){
		//  	char tmps_bk[1024];
		//  	cudaMemcpy(bkp,fdtdgpu.vz,nDim*sizeof(float),cudaMemcpyDeviceToHost);
		//  	sprintf(tmps_bk, "snap/snap_recFW_vz_it%d_%d_%d.dat",it,fdtdgpu.nxpml,fdtdgpu.nzpml);
		//  	write_1d_float_wb(bkp,nDim,tmps_bk);
		//  }	
		//  if(it%1000==0){
		//  	char tmps_bk[1024];
		//  	cudaMemcpy(bkp,fdtdgpu.vz_bk,nDim*sizeof(float),cudaMemcpyDeviceToHost);
		//  	sprintf(tmps_bk, "snap/snap_BW_vz_it%d_%d_%d.dat",it,fdtdgpu.nxpml,fdtdgpu.nzpml);
		//  	write_1d_float_wb(bkp,nDim,tmps_bk);
		//  }			 

	}

	cudaDeviceSynchronize();	

	// float *b_gradpp = new float[nx*nz];
	// cudaMemcpy(b_gradpp,fdtdgpu.b_numerator_pp,nx*nz*sizeof(float),cudaMemcpyDeviceToHost);
	// char tmps_bk[1024];
	// sprintf(tmps_bk, "snap/bgradpp%d_%d_%d_%d.dat",iter,sxnum,fdtdgpu.nx,fdtdgpu.nz);
	// write_1d_float_wb(b_gradpp,nx*nz,tmps_bk);
	// delete[] b_gradpp;


	delete[] record_scatter_x;
	delete[] record_scatter_z;
	delete[] record_residual_z;
	delete[] record_residual_x;

	if(!rbc){
    		cudaFree(GPU_theta_up);
	        cudaFree(GPU_theta_down);
	        cudaFree(GPU_theta_left);
	        cudaFree(GPU_theta_right);
 	        cudaFree(GPU_omega_up);
	        cudaFree(GPU_omega_down);
	        cudaFree(GPU_omega_left);
	        cudaFree(GPU_omega_right);
		if(cpu_mem){
		    delete[] CPU_theta_up;
		    delete[] CPU_theta_down;
		    delete[] CPU_omega_up;
		    delete[] CPU_omega_down;
		    delete[] CPU_theta_left;
		    delete[] CPU_theta_right;
		    delete[] CPU_omega_left;
		    delete[] CPU_omega_right;
		}
	}	


	if(!rbc){
    		cudaFree(GPU_theta_up2);
	        cudaFree(GPU_theta_down2);
	        cudaFree(GPU_theta_left2);
	        cudaFree(GPU_theta_right2);
 	        cudaFree(GPU_omega_up2);
	        cudaFree(GPU_omega_down2);
	        cudaFree(GPU_omega_left2);
	        cudaFree(GPU_omega_right2);
		if(cpu_mem){
		    delete[] CPU_theta_up2;
		    delete[] CPU_theta_down2;
		    delete[] CPU_omega_up2;
		    delete[] CPU_omega_down2;
		    delete[] CPU_theta_left2;
		    delete[] CPU_theta_right2;
		    delete[] CPU_omega_left2;
		    delete[] CPU_omega_right2;
		}
	}	

}




//////////////////////////////////////////////////////////////////////////
void CALCULATE_ALPHA(FD2DGPU_ELASTIC &fdtdgpu,const int sxnum,const int nDim,const int myid,float *record,float *record_x,int *gc, char *snapdir, bool rbc,bool cpu_mem,int ntt,int iter,\
	float *image_pp,float *image_ps,float *cpu_pp_grad_old,float *cpu_ps_grad_old,double &fenzi,double &fenmu){

	fdtdgpu.FD_ELASTIC_initial(nDim);
	fdtdgpu.BK_ELASTIC_initial(nDim);	
	fdtdgpu.LSRTM_initial(nDim);

	cudaMemcpy(fdtdgpu.image_pp_m,image_pp,fdtdgpu.allnx*fdtdgpu.allnz*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(fdtdgpu.image_ps_m,image_ps,fdtdgpu.allnx*fdtdgpu.allnz*sizeof(float),cudaMemcpyHostToDevice);

	fdtdgpu.buffer_image_extrapolation(fdtdgpu.cmin);
//	bool rbc = false;

	if(rbc){
		fdtdgpu.set_random_boundary();
	}
	else{
		fdtdgpu.calculate_elastic_damp_C();	
	}

	int direction = 1;
	int direction_rec = -1;					//TODO  1 OR -1
	bool forward = true;
	bool backward = false;	

	//  float* bkp = new float[nDim];
	//  memset(bkp,0,sizeof(float)*nDim);

	bool reconst = false;

	bool born = true;
	
	// cudaMemcpy(fdtdgpu.GPUgc,gc,fdtdgpu.ntr*sizeof(int),cudaMemcpyHostToDevice);

	int forward_born = 1;
	bool doublescatter = true;
//	Lm 
	for(int it=0; it<fdtdgpu.nt; ++it)    
    	{
        	fdtdgpu.elastic_rotation_addS_for(it,0); 
			fdtdgpu.born_modeling(direction,forward_born,it);
			// fdtdgpu.born_modeling(direction,5);			// 2nd scatter
			fdtdgpu.record_scatter(it,doublescatter);
    	}

	cudaDeviceSynchronize();

    // fdtdgpu.record_copytoGPU(record,record_x,0);
	float *record_scatter_z = new float[fdtdgpu.ntr*fdtdgpu.nt];
	float *record_scatter_x = new float[fdtdgpu.ntr*fdtdgpu.nt];
	float *record_residual_z = new float[fdtdgpu.ntr*fdtdgpu.nt];
	float *record_residual_x = new float[fdtdgpu.ntr*fdtdgpu.nt];			
	float scale_size = 1.0;
//	Lm + Lm2 - d 
	// char imagepath_pp[1024];
	// FILE *fpp = NULL;

	// cudaMemcpy(record_residual_z,fdtdgpu.GPUrecord_scatter_z,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),cudaMemcpyDeviceToHost);

	// if(sxnum==1||sxnum==20||sxnum==40){
	// sprintf(imagepath_pp, "%s/alpha_Lm_z_%d_%d_%d_%d.dat",snapdir,iter,fdtdgpu.ntr,fdtdgpu.nt,sxnum);		
	// fpp = fopen(imagepath_pp,"wb");
	// fwrite(record_residual_z,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),1,fpp);
	// fclose(fpp);
	// }	



	fdtdgpu.subtract(scale_size,doublescatter);					// Lm -d 

	// fdtdgpu.subtract_alpha();				// d - Lm 

	cudaMemcpy(record_residual_z,fdtdgpu.GPUrecord_scatter_z,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(record_residual_x,fdtdgpu.GPUrecord_scatter_x,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),cudaMemcpyDeviceToHost);	


	// if(sxnum==1||sxnum==20||sxnum==40){
	// sprintf(imagepath_pp, "%s/alpha_Lm-d_z_%d_%d_%d_%d.dat",snapdir,iter,fdtdgpu.ntr,fdtdgpu.nt,sxnum);		
	// fpp = fopen(imagepath_pp,"wb");
	// fwrite(record_residual_z,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),1,fpp);
	// fclose(fpp);
	// }	

	// float *record_z_cpu = new float[fdtdgpu.ntr*fdtdgpu.nt];
	// float *record_x_cpu = new float[fdtdgpu.ntr*fdtdgpu.nt];	

	// cudaMemcpy(record_scatter_z,fdtdgpu.GPUrecord_scatter_z,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),cudaMemcpyDeviceToHost);
	// cudaMemcpy(record_scatter_x,fdtdgpu.GPUrecord_scatter_x,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),cudaMemcpyDeviceToHost);	
	// cudaMemcpy(record_z_cpu,fdtdgpu.GPUrecord,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),cudaMemcpyDeviceToHost);
	// cudaMemcpy(record_x_cpu,fdtdgpu.GPUrecord_x,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),cudaMemcpyDeviceToHost);	

	// if(iter==0){

	// }
	// else{
	// 	for(int i=0;i<fdtdgpu.ntr;i++)
	// 	for(int j=0;j<fdtdgpu.nt;j++){
	// 		record_residual_x[i*fdtdgpu.nt+j] = ( record_x_cpu[i*fdtdgpu.nt+j] - record_scatter_x[i*fdtdgpu.nt+j]);
	// 		record_residual_z[i*fdtdgpu.nt+j] = ( record_z_cpu[i*fdtdgpu.nt+j] - record_scatter_z[i*fdtdgpu.nt+j]);											
	// 	}	
	// }


// alpha = gT*g/(gT*LTLg) = (LT*delta_d)T*(LT*delta_d)/() = delta_dT*(L*LT*delta_d)/()
	fdtdgpu.FD_ELASTIC_initial(nDim);
	fdtdgpu.BK_ELASTIC_initial(nDim);		
	fdtdgpu.LSRTM_initial(nDim);

	cudaMemcpy(fdtdgpu.image_pp_m,cpu_pp_grad_old,fdtdgpu.allnx*fdtdgpu.allnz*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(fdtdgpu.image_ps_m,cpu_ps_grad_old,fdtdgpu.allnx*fdtdgpu.allnz*sizeof(float),cudaMemcpyHostToDevice);
	fdtdgpu.buffer_image_extrapolation(fdtdgpu.cmin);
	// fdtdgpu.calculate_elastic_damp_C();		
	for(int it=0; it<fdtdgpu.nt; ++it)    
	{
		fdtdgpu.elastic_rotation_addS_for(it,0);
		fdtdgpu.born_modeling(direction,forward_born,it);
		// fdtdgpu.born_modeling(direction,5);			//2nd scatter
		fdtdgpu.record_scatter(it,doublescatter);
	}
	cudaDeviceSynchronize();
	// fdtdgpu.scatter_add(scale_size);			// Lm + Lm2	
	cudaMemcpy(record_scatter_z,fdtdgpu.GPUrecord_scatter_z,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(record_scatter_x,fdtdgpu.GPUrecord_scatter_x,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),cudaMemcpyDeviceToHost);	

	// if(sxnum==1||sxnum==20||sxnum==40){
	// sprintf(imagepath_pp, "%s/alpha_Lcg_z_%d_%d_%d_%d.dat",snapdir,iter,fdtdgpu.ntr,fdtdgpu.nt,sxnum);		
	// fpp = fopen(imagepath_pp,"wb");
	// fwrite(record_scatter_z,fdtdgpu.ntr*fdtdgpu.nt*sizeof(float),1,fpp);
	// fclose(fpp);
	// }	


	for(int i=0;i<fdtdgpu.ntr;i++)
	for(int j=0;j<fdtdgpu.nt;j++){
		fenmu += (record_scatter_z[i*fdtdgpu.nt+j]*record_scatter_z[i*fdtdgpu.nt+j] + record_scatter_x[i*fdtdgpu.nt+j]*record_scatter_x[i*fdtdgpu.nt+j]);
		fenzi += (record_scatter_z[i*fdtdgpu.nt+j]*record_residual_z[i*fdtdgpu.nt+j] + record_scatter_x[i*fdtdgpu.nt+j]*record_residual_x[i*fdtdgpu.nt+j]);									
	}	

	// std::cout<<"myid = "<<myid<<"  sx = "<<sxnum<<"  fenzi = "<<fenzi<<"  fenmu = "<<fenmu<<std::endl;


	delete[] record_scatter_x;
	delete[] record_scatter_z;
	delete[] record_residual_z;
	delete[] record_residual_x;

}
