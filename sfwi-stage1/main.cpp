//#define _CRT_SECURE_NO_WARNINGS
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include <math.h>
#include <sys/stat.h>
#define PI 3.14159265358979
#define FILE_NAME_MAX_LENGTH 256
    
#ifndef MAX 
#define MAX(x,y) ((x) > (y) ? (x) : (y))
#endif
#ifndef MIN
#define MIN(x,y) ((x) < (y) ? (x) : (y))
#endif

#include<omp.h>
#include "mpi.h"
#include "segy.h"
#include "FD2DGPU.cuh"


void normalize(float* matrix, int length, int tracenum) {
	float max = 0;
	for (int i = 0; i < length * tracenum; i++)
		max = (fabs(matrix[i]) > max) ? fabs(matrix[i]) : max;
	//normalized	
	for (int i = 0; i < length * tracenum; i++) {
		matrix[i] = matrix[i] / max;
	}
}


void normalize_all(float* matrix1,float* matrix2, int length, int tracenum) {
	float max = 0;
	for (int i = 0; i < length * tracenum; i++){
		max = (fabs(matrix1[i]) > max) ? fabs(matrix1[i]) : max;
    }
    for (int i = 0; i < length * tracenum; i++){
		max = (fabs(matrix2[i]) > max) ? fabs(matrix2[i]) : max;
    }
	//normalized	
	for (int i = 0; i < length * tracenum; i++) {
		matrix1[i] = matrix1[i] / max;
		matrix2[i] = matrix2[i] / max;        
	}
}



void cal_step(double misfit_step0,double misfit_step1,double misfit_step2,float step1,float step2,double &alpha) {

	float b = (step2 * step2 * misfit_step1 - step1 * step1 * misfit_step2 - misfit_step0 * (step2 * step2 - step1 * step1)) / (step1 * step2 * (step2 - step1));

	float a = (misfit_step1 - misfit_step0 - b * step1) / step1 / step1;

	alpha = -b / 2 / a;

}



template <typename T>
void cal_step2(T misfit_step0,T misfit_step1,T misfit_step2,float step1,float step2,double &alpha) {

	double fenzi = (step1*step1 - step2*step2)*misfit_step0 + (step2*step2)*misfit_step1 + ( - step1*step1)*misfit_step2; 

	double fenmu = (step1 - step2)*misfit_step0 + (step2)*misfit_step1 + ( - step1)*misfit_step2; 

	alpha = fenzi/fenmu/2.0f;

}


int    index_shot(char *fn, int *nt, float *dt, int *ns, int **table,int coordinate_scale=1)
{
       bhed   Bh;
       segy1  Th;
       Y_3200 Y3200;
       FILE   *fp ;
       int    cx_min_s, cx_max_s;
       int    ntr, pos;

       fp = fopen(fn,"rb");
       if(fp == NULL) {
          printf("Sorry,cann't open seismic file!\n");
          return 1;
       }
 
       fseek(fp, 0, SEEK_SET);
       fread(&Th, 240, 1, fp); //240 can be displaced by sizeof(segy1)
       *nt = (int)Th.ns;
       *dt = Th.dt/1000000.0;

       int TL = (*nt)*sizeof(float) ;
       
       fseek(fp, 0, SEEK_SET);
     
       *ns     = 0;
       pos     = 0;
       int sx0 = -999999;
       int sy0 = -999999;
       for( ; ; ){
          fread(&Th, 240, 1, fp);
          if(feof(fp)){
             int ns0 = *ns ;
             table[ns0-1][0] = ns0;
             table[ns0-1][1] = ntr;
             table[ns0-1][2] = sx0/coordinate_scale;
             table[ns0-1][3] = 0; //sy
             table[ns0-1][4] = cx_min_s/coordinate_scale;
             table[ns0-1][5] = 0;
             table[ns0-1][6] = cx_max_s/coordinate_scale;
             table[ns0-1][7] = 0;
             table[ns0-1][8] = pos - ntr;
             break;
          }

          int sx = Th.sx;
          int sy = 0;//2d,sy=0
          int gx = Th.gx;
          int gy = 0;
     
          int xmin = MIN(sx, gx);
          int xmax = MAX(sx, gx);

          if(sx != sx0){
              if(pos > 0){
                  int ns0 = *ns ;
                  table[ns0-1][0] = ns0;
                  table[ns0-1][1] = ntr;
                  table[ns0-1][2] = sx0/coordinate_scale;
                  table[ns0-1][3] = 0; //sy
                  table[ns0-1][4] = cx_min_s/coordinate_scale;
                  table[ns0-1][5] = 0;
                  table[ns0-1][6] = cx_max_s/coordinate_scale;
                  table[ns0-1][7] = 0;
                  table[ns0-1][8] = pos - ntr;
              }
              (*ns) ++;
              if((*ns)%50==0)printf(" %dth shot has been indexed!\n", (*ns));

              ntr = 1;

              sx0 = sx;
              sy0 = sy;
             
              cx_min_s = 99999999 ;
              cx_max_s = -999999 ;
          }else{

          }

          pos ++ ;
          if(xmin < cx_min_s) cx_min_s = xmin;
          if(xmax > cx_max_s) cx_max_s = xmax;      
  
          fseek(fp, TL, SEEK_CUR);
	
      } 
      fclose(fp); 
   
      return 0 ;
}

void   read_shot_gc_su(char *fn, long long int pos, int ntr, int nt,int *gc)
{
	int i;
    FILE   *fp ;
     float *dat = new float[ntr*nt];

    fp = fopen(fn,"rb");
    if(fp == NULL) {
            printf("Sorry,cann't open input seismic file!\n");
            exit(0);
    }
          
    int TL = 240 + nt*sizeof(float);
           
    fseek(fp, (long long int)TL*pos, SEEK_SET);

	segy1 Th;

    for(i=0; i<ntr; i++){
           fread(&Th, 240, 1, fp);
           gc[i]=Th.gx;               
           fread(&dat[i*nt], sizeof(float), nt, fp);
    }
    delete[] dat;
    fclose(fp);
}


void   read_shot_gather_su(char *fn, long long int pos, int ntr, int nt, float *dat,int *gc,int coordinate_scale=1)
{
	int i;
    FILE   *fp ;
     
    fp = fopen(fn,"rb");
    if(fp == NULL) {
            printf("Sorry,cann't open input seismic file!\n");
            exit(0);
    }
          
    int TL = 240 + nt*sizeof(float);
           
    fseek(fp, (long long int)TL*pos, SEEK_SET);

	segy1 Th;

    for(i=0; i<ntr; i++){
           fread(&Th, 240, 1, fp);
           gc[i]=Th.gx/coordinate_scale;               
           fread(&dat[i*nt], sizeof(float), nt, fp);
    }

    fclose(fp);
}

void   read_shot_gather_su2(char *fn, long long int pos, int ntr, int nt, float *dat,int *gc,int coordinate_scale=1)
{
	int i;
    FILE   *fp ;
    
	std::cout<<"==================enter differ read shot gather su======================="<<std::endl;
 
    fp = fopen(fn,"rb");
    if(fp == NULL) {
            printf("Sorry,cann't open input seismic file!\n");
            exit(0);
    }
          
    int TL = 240 + nt*sizeof(float);
           
    fseek(fp, (long long int)TL*pos, SEEK_SET);

	segy1 Th;

    for(i=0; i<ntr; i++){
           fread(&Th, 240, 1, fp);
           gc[i]=Th.gx/coordinate_scale;         
           fread(&dat[i*nt], sizeof(float), nt, fp);
    }

    fclose(fp);
}



void ricker_wave(float *w, int Tn, float dt, float FM)
{
	int t;
	float  Nk = PI*PI*FM*FM*dt*dt;
	int t0 = ceil(1.0 / (FM*dt));          
	for (t = 0; t<Tn; t++)
	{
		w[t] = (1.0 - 2.0*Nk*(t - t0)*(t - t0))*exp(-Nk*(t - t0)*(t - t0));
	}
}

// 		lsrtm   use   model   par
typedef struct{
    char fn1[1024];
    char fn2[1024];	
    char fn3[1024];	     
    char imagedir[1024];	    
    float *velp;
    float *vels;
    float *rho;      
    float *sou;
    float *record_z;
    float *record_x;
    int *gc;
    float dx,dz,dt;
    int minshot,maxshot,nx,nz,ns,nxpml,nzpml,allnx,allnz,scale,pml,nt,nop,ntr_pre;
    bool light,rbc,cpu_mem;
    int iointerval;
    int all_left;
} modelpar;


void init_modelparameters(modelpar *model,float dx,float dz,float dt,int minshot,int maxshot, int nx, int nz,\
            int ns,int nxpml,int nzpml,int allnx,int allnz,int scale,int pml,int nt,int nop,int ntr_pre,bool light,bool rbc,bool cpu_mem,int iointerval,int all_left){
    model->dx = dx; 
    model->dz = dz;
    model->dt = dt;
    model->minshot = minshot;
    model->maxshot = maxshot;
    model->nx = nx;
    model->nz = nz;    
    model->ns = ns;
    model->nxpml = nxpml;
    model->nzpml = nzpml;
    model->allnx = allnx;
    model->allnz = allnz;
    model->scale = scale;
    model->pml = pml;
    model->nt = nt;
    model->nop = nop;
    model->ntr_pre = ntr_pre;
    model->light  = light;
    model->rbc = rbc;
    model->cpu_mem = cpu_mem;
    model->iointerval = iointerval;
    model->all_left = all_left;
}
//

void all_iteration(int myid,int np,int sy,int gy,MPI_Status status,int **table,modelpar *model,float *image_pp,float *image_ps,float *pp_grad,float *ps_grad,float *illumination,int maxiter,int iter_round1,int record_left_in_v){

//	MPI_Status status;
//    	int myid,np;
	// MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	// MPI_Comm_size(MPI_COMM_WORLD,&np);


    float *b_pp_grad = new float[model->allnx*model->allnz];
    float *b_ps_grad = new float[model->allnx*model->allnz];
    float *b_pp_grad2 = new float[model->allnx*model->allnz];
    float *b_ps_grad2 = new float[model->allnx*model->allnz];

    float *sum_pp_grad = new float[model->allnx*model->allnz];
    float *sum_ps_grad = new float[model->allnx*model->allnz];
    if(myid!=0){
        cudaSetDevice((myid-1)%8);			
        // FD2DGPU_ELASTIC image2delastic(model->sou,model->dx,model->dz,model->dt,model->nxpml,model->nzpml,model->allnx,model->allnz,model->scale,model->pml,model->nt,model->nop,model->ntr_pre);	
        // image2delastic.GPUbufferVPVS(model->velp,model->vels);
    }

    int *members = new int[np-1];
    for(int i=0;i<np-1;i++){
        members[i]=i+1;
    }

    MPI_Group group_world,group_new;
    MPI_Comm groupcomm;
    MPI_Comm_group(MPI_COMM_WORLD,&group_world);
    MPI_Group_incl(group_world, np-1, members, &group_new);
    MPI_Comm_create(MPI_COMM_WORLD, group_new, &groupcomm);

    // int id_ingroup,np_ingroup;
	// MPI_Comm_rank(groupcomm,&id_ingroup);
	// MPI_Comm_size(groupcomm,&np_ingroup);

    // if(myid!=0){
    //     std::cout<<"myid = "<<myid<<std::endl;    //<<"\t id_ingroup = "<<id_ingroup<<"\t np_ingroup = "<<np_ingroup<<std::endl;
    // }

	int ip;
	int send[9],recv[9];
	int nsend,ntask;
	ntask = model->ns;   

    double misfit =0;

    float misfit_ot = 0;

    double max_misfit =0;
    double rms_misfit =0;

    double fenzi =0;
    double fenmu =0;        

    double misfit_step0 = 0;
    double misfit_step1 = 0;
    double misfit_step2 = 0;

    float misfit_step0_ot = 0;
    float misfit_step1_ot = 0;
    float misfit_step2_ot = 0;

    float step1 = 30;
    float step2 = 150;

    double beta_fenzi =0;
    double beta_fenmu =0;
    double beta =0;

    float *pp_cg = new float[model->allnx*model->allnz]{};
    float *ps_cg = new float[model->allnx*model->allnz]{};

    float *pp_cg_old = new float[model->allnx*model->allnz]{};
    float *ps_cg_old = new float[model->allnx*model->allnz]{};

    float array_misfit[2][maxiter];

    double alpha =0;

// use for hessian-vector

    float *hessian_p_pp = new float[model->allnx*model->allnz]{};
    float *hessian_p_ps = new float[model->allnx*model->allnz]{};    

    float *hessian_Hp_pp = new float[model->allnx*model->allnz]{};
    float *hessian_Hp_ps = new float[model->allnx*model->allnz]{};  

    float *hessian_r_pp = new float[model->allnx*model->allnz]{};
    float *hessian_r_ps = new float[model->allnx*model->allnz]{};    

    float *hessian_delta_pp = new float[model->allnx*model->allnz]{};
    float *hessian_delta_ps = new float[model->allnx*model->allnz]{};    

    float *hessian_grad_pp = new float[model->allnx*model->allnz]{};
    float *hessian_grad_ps = new float[model->allnx*model->allnz]{};    

//
    int max_inner = 0;
    bool cut_grad = true;
    int cut_layer = 26;
    int hessian_cut_layer = cut_layer;
    int hessian_decrease_layer = model->allnz;

    bool update_mb = false;
    bool update_mp = true;

	if(myid==0)
	{
    for(int iter=0;iter<maxiter;iter++)
    {
        printf("======================================================\n\n");      
        printf(" iter           : %d\n",iter);
        printf(" maxiter        : %d\n",maxiter);
        printf(" process        : %f%\n\n",(float)iter/maxiter*100);
        printf("======================================================\n\n");      
		nsend = 0;
		for(int i=0;i<ntask+np-1;i++)
		{
			MPI_Recv(recv,9,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			ip = status.MPI_SOURCE;
			if(i<model->ns)
			{
                         send[0] = table[i][0];
                         send[1] = table[i][1];
                         send[2] = table[i][2];
                         send[3] = table[i][3];
                         send[4] = table[i][4];
                         send[5] = table[i][5];
                         send[6] = table[i][6];
                         send[7] = table[i][7];
                         send[8] = table[i][8];
			}
			else
			{
				// printf("shotnum = %d\n",i);				
				send[0] = 0;
			}
			
			MPI_Send(send,9,MPI_INT,ip,99,MPI_COMM_WORLD);
			nsend = nsend+1;
			// if(i<ntask)printf("Calculating Gradient. Send No.=%d. Shot to Processor %d\n",send[0],ip);
//			fflush(stdout);
		}
      

        bool update_mp_0 = false;

		MPI_Recv(&update_mp_0,1,MPI_C_BOOL,1,77,MPI_COMM_WORLD,&status);

//calculate hessian-vector      
        if(iter>=0)
        // if(iter>=0&&update_mp_0)
        {
            for(int inner_loop=0;inner_loop<max_inner;inner_loop++)
            {
                nsend = 0;
                for(int i=0;i<ntask+np-1;i++)
                {
                    MPI_Recv(recv,9,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
                    ip = status.MPI_SOURCE;
                    if(i<model->ns)
                    {
                                send[0] = table[i][0];
                                send[1] = table[i][1];
                                send[2] = table[i][2];
                                send[3] = table[i][3];
                                send[4] = table[i][4];
                                send[5] = table[i][5];
                                send[6] = table[i][6];
                                send[7] = table[i][7];
                                send[8] = table[i][8];
                    }
                    else
                    {
                        // printf("shotnum = %d\n",i);				
                        send[0] = 0;
                    }
                    
                    MPI_Send(send,9,MPI_INT,ip,99,MPI_COMM_WORLD);
                    nsend = nsend+1;
                    // if(i<ntask)printf("Update image. Send No.=%d. Shot to Processor %d\n",send[0],ip);
        //			fflush(stdout);
                }
            }
        }



//calculate alpha & update m
		nsend = 0;
		for(int i=0;i<ntask+np-1;i++)
		{
			MPI_Recv(recv,9,MPI_INT,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);
			ip = status.MPI_SOURCE;
			if(i<model->ns)
			{
                         send[0] = table[i][0];
                         send[1] = table[i][1];
                         send[2] = table[i][2];
                         send[3] = table[i][3];
                         send[4] = table[i][4];
                         send[5] = table[i][5];
                         send[6] = table[i][6];
                         send[7] = table[i][7];
                         send[8] = table[i][8];
			}
			else
			{
				// printf("shotnum = %d\n",i);				
				send[0] = 0;
			}
			
			MPI_Send(send,9,MPI_INT,ip,99,MPI_COMM_WORLD);
			nsend = nsend+1;
			// if(i<ntask)printf("Update image. Send No.=%d. Shot to Processor %d\n",send[0],ip);
//			fflush(stdout);
		}
	MPI_Barrier(MPI_COMM_WORLD);
    }
	}
	else    
	{
    // cudaSetDevice((myid-1)%8);			
    FD2DGPU_ELASTIC image2delastic(model->sou,model->dx,model->dz,model->dt,model->nxpml,model->nzpml,model->allnx,model->allnz,model->scale,model->pml,model->nt,model->nop,model->ntr_pre);	
    image2delastic.GPUbufferVPVS(model->velp,model->vels,model->rho);
    for(int iter=0;iter<maxiter;iter++)
    {
        misfit =0;

        misfit_ot =0;

        beta_fenzi =0;
        memset(pp_grad,0,sizeof(float)*model->allnx*model->allnz);        
        memset(ps_grad,0,sizeof(float)*model->allnx*model->allnz); 
        memset(illumination,0,sizeof(float)*model->allnx*model->allnz);                     
     
        memset(b_pp_grad,0,sizeof(float)*model->allnx*model->allnz);        
        memset(b_ps_grad,0,sizeof(float)*model->allnx*model->allnz); 
        memset(b_pp_grad2,0,sizeof(float)*model->allnx*model->allnz);        
        memset(b_ps_grad2,0,sizeof(float)*model->allnx*model->allnz); 

        image2delastic.GPUbufferM(image_pp,image_ps); 
        image2delastic.GPUbufferVPVS(model->velp,model->vels,model->rho);

		MPI_Send(send,9,MPI_INT,0,0,MPI_COMM_WORLD);
		for(;;)
		{
			MPI_Recv(recv,9,MPI_INT,0,99,MPI_COMM_WORLD,&status);

            int sno = recv[0];
            int ntr = recv[1];
            int scx = recv[2];
            int scy = recv[3];
            int cx_min_s = recv[4];
            int cy_min_s = recv[5];
            int cx_max_s = recv[6];
            int cy_max_s = recv[7];
            int pos = recv[8];

			if(sno == 0)
			{
				// printf("myid=%d,calculating gradient finished,waiting...\n",myid);	
				break;
			}

            if( sno<model->minshot||sno>model->maxshot ) {
                // printf(" the %dth shot is out of imaging range\n", sno);
                // fflush(stdout);
			    MPI_Send(send,9,MPI_INT,0,myid,MPI_COMM_WORLD); 
                continue;               
            }

			printf("Calculating Gradient,  %d Shot in Processor %d\n",sno,myid);

		    int   ngx_left  = cx_min_s;
			int   ngx_right = cx_max_s;
			int   nx_s = (cx_max_s - cx_min_s)/model->dx + 1 ;
            // printf("all_left=%d,ngx_left=%d,ngx_right=%d\n nx_s=%d,pos=%d,ntr=%d\n",model->all_left,ngx_left,ngx_right,nx_s,pos,ntr);
	        int delta_left =int((ngx_left-model->all_left)/model->dx);

            if(ntr==image2delastic.ntr&&ntr==model->ntr_pre){
                read_shot_gather_su(model->fn1, pos, ntr, model->nt, model->record_z, model->gc);
                read_shot_gather_su(model->fn2, pos, ntr, model->nt, model->record_x, model->gc);         
                image2delastic.record_copytoGPU(model->record_z,model->record_x,model->gc);
            }
            else{
                printf("Trace numbers differ from preload ntr_pre in this shot. Need reallocate.\n");
		        image2delastic.ntr = ntr;
                delete[] model->record_z;
                delete[] model->record_x;
                delete[] model->gc;                
                model->record_z = new float[model->nt*ntr]{};
                model->record_x = new float[model->nt*ntr]{};                
                model->gc = new int[ntr]{};
                read_shot_gather_su2(model->fn1, pos, ntr, model->nt, model->record_z, model->gc);  
                read_shot_gather_su2(model->fn2, pos, ntr, model->nt, model->record_x, model->gc); 		                 		
                image2delastic.record_copytoGPU(model->record_z,model->record_x,model->gc);
		        std::cout<<"reallocate completed"<<std::endl;
            }
            if(nx_s==model->nx){
			    image2delastic.bufferVpVsHtoD(delta_left+record_left_in_v);
            }
            else{
                printf("Nx differ from preload ntr_pre in this shot. Need reallocate.\n");
                //TODO reallocate space and memory copy
			    image2delastic.bufferVpVsHtoD(delta_left+record_left_in_v);
            }

            image2delastic.delta_left = delta_left;
            image2delastic.record_left_in_v = record_left_in_v;                        
	        image2delastic.isx = model->pml+(int)((scx-ngx_left)/model->dx);
            image2delastic.isz = model->pml + sy;
            image2delastic.igz = model->pml + gy;
            image2delastic.cmin = delta_left+record_left_in_v;          //most left position(grid) in velocity/image model 
            image2delastic.ngx_left = ngx_left;                         //most left positon   
			int Dim = model->nxpml*model->nzpml;
            // printf("scx = %d,isx = %d\n",scx,image2delastic.isx);
			// printf("source num=%d\n",sno);
		    // printf("gc1=%d,gc2=%d\n",model->gc[0],model->gc[ntr-1]);

			IMAGE_SINGLESHOT(image2delastic,sno,Dim,myid,model->record_z,model->record_x,model->gc,model->rbc,model->cpu_mem,model->iointerval,iter,iter_round1,maxiter,model->imagedir,image_pp,image_ps,pp_grad,ps_grad,misfit,&misfit_ot);

			MPI_Send(send,9,MPI_INT,0,myid,MPI_COMM_WORLD);
		}
        image2delastic.imagebuffer_resettozero(pp_grad,ps_grad,illumination,b_pp_grad,b_ps_grad,b_pp_grad2,b_ps_grad2);             //TODO:bug exist!
		printf("myid=%d,calculating gradient finished,waiting...\n",myid);	        
        MPI_Barrier(groupcomm);
        if(myid==1){
        MPI_Reduce(MPI_IN_PLACE, &pp_grad[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);
        MPI_Reduce(MPI_IN_PLACE, &ps_grad[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);	
        MPI_Reduce(MPI_IN_PLACE, &illumination[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);
        MPI_Reduce(MPI_IN_PLACE, &misfit, 1, MPI_DOUBLE, MPI_SUM, 0, groupcomm);        
        MPI_Reduce(MPI_IN_PLACE, &misfit_ot, 1, MPI_FLOAT, MPI_SUM, 0, groupcomm);    



        MPI_Reduce(MPI_IN_PLACE, &b_pp_grad[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);
        MPI_Reduce(MPI_IN_PLACE, &b_ps_grad[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);	 

        MPI_Reduce(MPI_IN_PLACE, &b_pp_grad2[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);
        MPI_Reduce(MPI_IN_PLACE, &b_ps_grad2[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);	                
        }
        else{
        MPI_Reduce(&pp_grad[0], &pp_grad[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);
        MPI_Reduce(&ps_grad[0], &ps_grad[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);	
        MPI_Reduce(&illumination[0], &illumination[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);        
        MPI_Reduce(&misfit, &misfit, 1, MPI_DOUBLE, MPI_SUM, 0, groupcomm);           
        MPI_Reduce(&misfit_ot, &misfit_ot, 1, MPI_FLOAT, MPI_SUM, 0, groupcomm);      


        MPI_Reduce(&b_pp_grad[0], &b_pp_grad[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);
        MPI_Reduce(&b_ps_grad[0], &b_ps_grad[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);	   

        MPI_Reduce(&b_pp_grad2[0], &b_pp_grad2[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);
        MPI_Reduce(&b_ps_grad2[0], &b_ps_grad2[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);	                
        }
		if(myid==1){
            std::cout<<"iter = "<<iter<<"\tsum of misfit = "<<misfit<<std::endl;
            if(iter==0){
                max_misfit = misfit;
            }
            rms_misfit = misfit/max_misfit;



            std::cout<<"iter = "<<iter<<"\nnormalized  misfit = "<<rms_misfit<<std::endl;
            array_misfit[0][iter] = misfit;      
            array_misfit[1][iter] = rms_misfit;            


            char misfitdir[1024];
            sprintf(misfitdir, "%s/misfit.txt",model->imagedir);            
            FILE *fpout = fopen(misfitdir,"a");
            fprintf(fpout,"i = %d,misfit = %4.3e,rms = %f\n",iter,array_misfit[0][iter],array_misfit[1][iter]);
            fclose(fpout);

            bool illum = true;

            for(int i=0;i<model->allnx;i++)
                for(int j=0;j<model->allnz;j++)
                {

                    sum_pp_grad[i*model->allnz+j] = pp_grad[i*model->allnz+j];
                    sum_ps_grad[i*model->allnz+j] = ps_grad[i*model->allnz+j];                                 

                    if(illum){
                        sum_pp_grad[i*model->allnz+j] = sum_pp_grad[i*model->allnz+j]/illumination[i*model->allnz+j];
                        sum_ps_grad[i*model->allnz+j] = sum_ps_grad[i*model->allnz+j]/illumination[i*model->allnz+j];                                                                                                     
                    }
                }


            for(int i=0;i<model->allnx;i++)
                for(int j=0;j<model->allnz;j++)
                {
                    if(cut_grad&&j<cut_layer){
                        sum_pp_grad[i*model->allnz+j] = 0;
                        sum_ps_grad[i*model->allnz+j] = 0;                                 
                    }
                    else{
                        // sum_pp_grad[i*model->allnz+j] *= 1e10;                    // float cannot handle small number like e-50 and lead to zero
                        // sum_ps_grad[i*model->allnz+j] *= 1e10;                            
                    }
                }            


            for(int i=0;i<model->allnx;i++)
                for(int j=0;j<model->allnz;j++)
                {
                    pp_cg[i*model->allnz+j] =  sum_pp_grad[i*model->allnz+j];
                    ps_cg[i*model->allnz+j] =  sum_ps_grad[i*model->allnz+j];     

                    beta_fenzi += pp_grad[i*model->allnz+j]*sum_pp_grad[i*model->allnz+j] + ps_grad[i*model->allnz+j]*sum_ps_grad[i*model->allnz+j];      
                    
                    // beta_fenzi += sum_pp_grad[i*model->allnz+j]*sum_pp_grad[i*model->allnz+j] + sum_ps_grad[i*model->allnz+j]*sum_ps_grad[i*model->allnz+j];      

                }


            if(iter==0){
                beta = 0;
            }
            else{
                beta = beta_fenzi/beta_fenmu;
            }

            std::cout<<"iter = "<<iter<<"\tbeta = "<<beta<<std::endl;

            #pragma omp parallel for num_threads(16)
            for(int i=0;i<model->allnx;i++)
                for(int j=0;j<model->allnz;j++)
                {
                    pp_cg[i*model->allnz+j] = pp_cg[i*model->allnz+j] + beta*pp_cg_old[i*model->allnz+j];                 
                    ps_cg[i*model->allnz+j] = ps_cg[i*model->allnz+j] + beta*ps_cg_old[i*model->allnz+j];                     
                }

            memcpy(pp_cg_old,pp_cg,model->allnx*model->allnz*sizeof(float));
            memcpy(ps_cg_old,ps_cg,model->allnx*model->allnz*sizeof(float));

            beta_fenmu = beta_fenzi;

            char imagepath_pp[1024];
            FILE *fpp = NULL;            
            sprintf(imagepath_pp, "%s/sum_grad_pp_%d_%d_%d.dat",model->imagedir,iter,model->allnx,model->allnz);
            fpp = fopen(imagepath_pp,"wb");
            fwrite(pp_grad,model->allnx*model->allnz*sizeof(float),1,fpp);
            fclose(fpp);  

            sprintf(imagepath_pp, "%s/sum_grad_ps_%d_%d_%d.dat",model->imagedir,iter,model->allnx,model->allnz);
            fpp = fopen(imagepath_pp,"wb");
            fwrite(ps_grad,model->allnx*model->allnz*sizeof(float),1,fpp);
            fclose(fpp);  
 
            sprintf(imagepath_pp, "%s/sum_cg_pp_%d_%d_%d.dat",model->imagedir,iter,model->allnx,model->allnz);
            fpp = fopen(imagepath_pp,"wb");
            fwrite(pp_cg,model->allnx*model->allnz*sizeof(float),1,fpp);
            fclose(fpp);  

		    MPI_Send(&update_mp,1,MPI_C_BOOL,0,77,MPI_COMM_WORLD);

        }

        MPI_Bcast(pp_cg, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);
        MPI_Bcast(ps_cg, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);           

        MPI_Bcast(sum_pp_grad, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);
        MPI_Bcast(sum_ps_grad, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);           

        MPI_Bcast(&update_mb, 1, MPI_C_BOOL, 0, groupcomm);
        MPI_Bcast(&update_mp, 1, MPI_C_BOOL, 0, groupcomm);   

        MPI_Barrier(groupcomm);

        if(iter>=0)
        // if(iter>=0&&update_mp)
        {

//TODO:  use origin gradient 
            // memcpy(hessian_r_pp,pp_cg,model->allnx*model->allnz*sizeof(float));
            // memcpy(hessian_r_ps,ps_cg,model->allnx*model->allnz*sizeof(float));
  
            memset(hessian_r_pp,0,sizeof(float)*model->allnx*model->allnz);        
            memset(hessian_r_ps,0,sizeof(float)*model->allnx*model->allnz); 
            memset(hessian_p_pp,0,sizeof(float)*model->allnx*model->allnz);        
            memset(hessian_p_ps,0,sizeof(float)*model->allnx*model->allnz); 

            memset(hessian_Hp_pp,0,sizeof(float)*model->allnx*model->allnz);        
            memset(hessian_Hp_ps,0,sizeof(float)*model->allnx*model->allnz); 
            memset(hessian_delta_pp,0,sizeof(float)*model->allnx*model->allnz);        
            memset(hessian_delta_ps,0,sizeof(float)*model->allnx*model->allnz); 
            memset(hessian_grad_pp,0,sizeof(float)*model->allnx*model->allnz);        
            memset(hessian_grad_ps,0,sizeof(float)*model->allnx*model->allnz); 


//SD
            // memcpy(hessian_r_pp,sum_pp_grad,model->allnx*model->allnz*sizeof(float));
            // memcpy(hessian_r_ps,sum_ps_grad,model->allnx*model->allnz*sizeof(float));

//CG
            memcpy(hessian_r_pp,pp_cg,model->allnx*model->allnz*sizeof(float));
            memcpy(hessian_r_ps,ps_cg,model->allnx*model->allnz*sizeof(float));

            normalize_all(hessian_r_pp,hessian_r_ps,model->allnz,model->allnx);     // normalize grad 

            for(int i=0;i<model->allnx*model->allnz;i++)
            {
                // hessian_r_pp[i] *= 1e10;                    // float cannot handle small number like e-50 and lead to zero
                // hessian_r_ps[i] *= 1e10;    

                hessian_p_pp[i] =  -hessian_r_pp[i];   
                hessian_p_ps[i] =  -hessian_r_ps[i];                               
            }



            bool sd = true;
            if(sd){
//SD                
                // memcpy(hessian_grad_pp,sum_pp_grad,model->allnx*model->allnz*sizeof(float));
                // memcpy(hessian_grad_ps,sum_ps_grad,model->allnx*model->allnz*sizeof(float));  
//CG
                memcpy(hessian_grad_pp,pp_cg,model->allnx*model->allnz*sizeof(float));
                memcpy(hessian_grad_ps,ps_cg,model->allnx*model->allnz*sizeof(float));  

                for(int i=0;i<model->allnx*model->allnz;i++)
                {
                    hessian_delta_pp[i] =  -hessian_grad_pp[i];   
                    hessian_delta_ps[i] =  -hessian_grad_ps[i];                               
                }                              
                normalize_all(hessian_delta_pp,hessian_delta_ps,model->allnz,model->allnx);        
                normalize_all(hessian_grad_pp,hessian_grad_ps,model->allnz,model->allnx);                                
            }


//calculate hessian-vector
            for(int inner_loop=0;inner_loop<max_inner;inner_loop++)
            {
                MPI_Barrier(groupcomm);
                MPI_Send(send,9,MPI_INT,0,0,MPI_COMM_WORLD);
                for(;;)
                {
                    MPI_Recv(recv,9,MPI_INT,0,99,MPI_COMM_WORLD,&status);

                    int sno = recv[0];
                    int ntr = recv[1];
                    int scx = recv[2];
                    int scy = recv[3];
                    int cx_min_s = recv[4];
                    int cy_min_s = recv[5];
                    int cx_max_s = recv[6];
                    int cy_max_s = recv[7];
                    int pos = recv[8];

                    if(sno == 0)
                    {
                        // printf("myid=%d,update image finished,waiting...\n",myid);	
                        break;
                    }

                    if( sno<model->minshot||sno>model->maxshot ) {
                        // printf(" the %dth shot is out of imaging range\n", sno);
                        // fflush(stdout);
                        MPI_Send(send,9,MPI_INT,0,myid,MPI_COMM_WORLD); 
                        continue;               
                    }

                    // printf("Update image, %d Shot in Processor %d\n",sno,myid);

                    int   ngx_left  = cx_min_s;
                    int   ngx_right = cx_max_s;
                    int   nx_s = (cx_max_s - cx_min_s)/model->dx + 1 ;
                    // printf("all_left=%d,ngx_left=%d,ngx_right=%d\n nx_s=%d,pos=%d,ntr=%d\n",model->all_left,ngx_left,ngx_right,nx_s,pos,ntr);
                    int delta_left =int((ngx_left-model->all_left)/model->dx);

                    if(ntr==image2delastic.ntr&&ntr==model->ntr_pre){
                        read_shot_gather_su(model->fn1, pos, ntr, model->nt, model->record_z, model->gc);
                        read_shot_gather_su(model->fn2, pos, ntr, model->nt, model->record_x, model->gc);             
                        image2delastic.record_copytoGPU(model->record_z,model->record_x,model->gc);
                    }
                    else{
                        printf("Trace numbers differ from preload ntr_pre in this shot. Need reallocate.\n");
                        image2delastic.ntr = ntr;
                        delete[] model->record_z;
                        delete[] model->record_x;
                        delete[] model->gc;                
                        model->record_z = new float[model->nt*ntr]{};
                        model->record_x = new float[model->nt*ntr]{};                
                        model->gc = new int[ntr]{};
                        read_shot_gather_su2(model->fn1, pos, ntr, model->nt, model->record_z, model->gc);  
                        read_shot_gather_su2(model->fn2, pos, ntr, model->nt, model->record_x, model->gc); 				
                        image2delastic.record_copytoGPU(model->record_z,model->record_x,model->gc);
                        std::cout<<"reallocate completed"<<std::endl;
                    }
                    if(nx_s==model->nx){
                        image2delastic.bufferVpVsHtoD(delta_left+record_left_in_v);
                    }
                    else{
                        printf("Nx differ from preload ntr_pre in this shot. Need reallocate.\n");
                        //TODO reallocate space and memory copy
                        image2delastic.bufferVpVsHtoD(delta_left+record_left_in_v);
                    }
            
                    image2delastic.delta_left = delta_left;
                    image2delastic.record_left_in_v = record_left_in_v;            
                    image2delastic.isx = model->pml+(int)((scx-ngx_left)/model->dx);
                    image2delastic.isz = model->pml + sy;
                    image2delastic.igz = model->pml + gy;
                    image2delastic.cmin = delta_left+record_left_in_v;          //most left position(grid) in velocity/image model 
                    image2delastic.ngx_left = ngx_left;                         //most left positon   
                    int Dim = model->nxpml*model->nzpml;
                    // printf("scx = %d,isx = %d\n",scx,image2delastic.isx);
                    // printf("source num=%d\n",sno);
                    // printf("gc1=%d,gc2=%d\n",model->gc[0],model->gc[ntr-1]);      

//TODO: Hessian of mb is different of mp.  Need to code another hessian-vector                   
                if(update_mp){
                    // ELSRTM_CALCULATE_Hessian_vector_misfit(image2delastic,sno,Dim,myid,model->record_z,model->record_x,model->gc,model->rbc,model->cpu_mem,model->iointerval,iter,hessian_p_pp,hessian_p_ps,hessian_delta_pp,hessian_delta_ps);
                    ELSRTM_CALCULATE_Hessian_vector(image2delastic,sno,Dim,myid,model->record_z,model->record_x,model->gc,model->rbc,model->cpu_mem,model->iointerval,iter,hessian_p_pp,hessian_p_ps);                    
                }
                else{
                    // ELSRTM_CALCULATE_Hessian_vector_misfit(image2delastic,sno,Dim,myid,model->record_z,model->record_x,model->gc,model->rbc,model->cpu_mem,model->iointerval,iter,hessian_p_pp,hessian_p_ps,hessian_delta_pp,hessian_delta_ps);     
                    ELSRTM_CALCULATE_Hessian_vector(image2delastic,sno,Dim,myid,model->record_z,model->record_x,model->gc,model->rbc,model->cpu_mem,model->iointerval,iter,hessian_p_pp,hessian_p_ps);                                        
                    // ELSRTM_CALCULATE_Hessian_vector_mb(image2delastic,sno,Dim,myid,model->record_z,model->record_x,model->gc,model->rbc,model->cpu_mem,model->iointerval,iter,hessian_p_pp,hessian_p_ps,model->imagedir);                    
                }

//TODO: use origin grad not cg_grad
                    MPI_Send(send,9,MPI_INT,0,myid,MPI_COMM_WORLD);
                }  
                image2delastic.imagebuffer_resettozero(hessian_Hp_pp,hessian_Hp_ps,illumination,b_pp_grad,b_ps_grad,b_pp_grad2,b_ps_grad2);             //TODO:bug exist!
                printf("myid=%d,hessian-vector finish %d-%d, waiting...\n",myid,iter,inner_loop);	        
                MPI_Barrier(groupcomm);
                if(myid==1){
                    MPI_Reduce(MPI_IN_PLACE, &hessian_Hp_pp[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);
                    MPI_Reduce(MPI_IN_PLACE, &hessian_Hp_ps[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);	

                    MPI_Reduce(MPI_IN_PLACE, &b_pp_grad[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);           //b_pp_grad ->  H * delta_m
                    MPI_Reduce(MPI_IN_PLACE, &b_ps_grad[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);	                              
                }
                else{
                    MPI_Reduce(&hessian_Hp_pp[0], &hessian_Hp_pp[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);
                    MPI_Reduce(&hessian_Hp_ps[0], &hessian_Hp_ps[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);	    
                                   

                    MPI_Reduce(&b_pp_grad[0], &b_pp_grad[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);
                    MPI_Reduce(&b_ps_grad[0], &b_ps_grad[0], model->allnx*model->allnz, MPI_FLOAT, MPI_SUM, 0, groupcomm);	                                    
                }

                MPI_Barrier(groupcomm);

                if(myid==1){

                    double hessian_alpha_fenzi = 0;
                    double hessian_alpha_fenmu = 0;

                    double hessian_beta_fenzi = 0;
                    double hessian_beta_fenmu = 0;

                    double hessian_alpha = 0;
                    double hessian_beta = 0;

                    for(int i=0;i<model->allnx;i++)
                        for(int j=0;j<model->allnz;j++)
                        {

                            if(cut_grad&&j<cut_layer){
                                hessian_Hp_pp[i*model->allnz+j] = 0;
                                hessian_Hp_ps[i*model->allnz+j] = 0;                                                               
                            }


                            hessian_alpha_fenzi += hessian_r_pp[i*model->allnz+j]*hessian_r_pp[i*model->allnz+j] + hessian_r_ps[i*model->allnz+j]*hessian_r_ps[i*model->allnz+j];  

                            hessian_alpha_fenmu += hessian_p_pp[i*model->allnz+j]*hessian_Hp_pp[i*model->allnz+j] + hessian_p_ps[i*model->allnz+j]*hessian_Hp_ps[i*model->allnz+j];                                             
                        }

                    char misfitdir[1024];

                    if(inner_loop==0){
                        sprintf(misfitdir, "%s/misfit.txt",model->imagedir);            
                        FILE *fpout = fopen(misfitdir,"a");
                        fprintf(fpout,"i = %d,inner = %d,hessian_misfit = %5.3e\n",iter,inner_loop,hessian_alpha_fenzi);
                        fclose(fpout);
                    }

                    hessian_alpha = hessian_alpha_fenzi/hessian_alpha_fenmu;

                    hessian_beta_fenmu = hessian_alpha_fenzi;

                    for(int i=0;i<model->allnx;i++)
                        for(int j=0;j<model->allnz;j++)
                        {
                            hessian_delta_pp[i*model->allnz+j] += hessian_alpha*hessian_p_pp[i*model->allnz+j];  
                            hessian_delta_ps[i*model->allnz+j] += hessian_alpha*hessian_p_ps[i*model->allnz+j];    

                            hessian_grad_pp[i*model->allnz+j] = -hessian_delta_pp[i*model->allnz+j];  
                            hessian_grad_ps[i*model->allnz+j] = -hessian_delta_ps[i*model->allnz+j];    

                            hessian_r_pp[i*model->allnz+j] += hessian_alpha*hessian_Hp_pp[i*model->allnz+j];  
                            hessian_r_ps[i*model->allnz+j] += hessian_alpha*hessian_Hp_ps[i*model->allnz+j];       

                            hessian_beta_fenzi += hessian_r_pp[i*model->allnz+j]*hessian_r_pp[i*model->allnz+j] + hessian_r_ps[i*model->allnz+j]*hessian_r_ps[i*model->allnz+j];                                                                
                        }                    

                    hessian_beta = hessian_beta_fenzi/hessian_beta_fenmu;

                    for(int i=0;i<model->allnx;i++)
                        for(int j=0;j<model->allnz;j++)
                        {
                            hessian_p_pp[i*model->allnz+j] = -hessian_r_pp[i*model->allnz+j] + hessian_beta*hessian_p_pp[i*model->allnz+j];  
                            hessian_p_ps[i*model->allnz+j] = -hessian_r_ps[i*model->allnz+j] + hessian_beta*hessian_p_ps[i*model->allnz+j];                                                                
                        }  

                    double hessian_misfit = 0;


                    sprintf(misfitdir, "%s/misfit.txt",model->imagedir);            
                    FILE *fpout = fopen(misfitdir,"a");
                    fprintf(fpout,"i = %d,inner = %d,hessian_misfit = %5.3e\n",iter,inner_loop+1,hessian_beta_fenzi);
                    fclose(fpout);

                    std::cout<<"hessian_misfit = "<<hessian_beta_fenzi<<"  hessian alpha = "<<hessian_alpha<<"  hessian beta = "<<hessian_beta<<std::endl;


                    char imagepath_pp[1024];
                    sprintf(imagepath_pp, "%s/hessian_delta_pp_%d_%d_%d_%d.dat",model->imagedir,iter,model->allnx,model->allnz,inner_loop);
                    FILE *fpp = NULL;
                    fpp = fopen(imagepath_pp,"wb");
                    fwrite(hessian_delta_pp,model->allnx*model->allnz*sizeof(float),1,fpp);
                    fclose(fpp);

                    sprintf(imagepath_pp, "%s/hessian_delta_ps_%d_%d_%d_%d.dat",model->imagedir,iter,model->allnx,model->allnz,inner_loop);
                    fpp = fopen(imagepath_pp,"wb");
                    fwrite(hessian_delta_ps,model->allnx*model->allnz*sizeof(float),1,fpp);
                    fclose(fpp);    

                }

                MPI_Bcast(hessian_p_pp, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);
                MPI_Bcast(hessian_p_ps, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);         

                MPI_Bcast(hessian_delta_pp, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);
                MPI_Bcast(hessian_delta_ps, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);                  

                MPI_Barrier(groupcomm);

            }       // hessian-vector inner loop
          

            if(update_mb||update_mp)
            {
                normalize_all(hessian_delta_pp,hessian_delta_ps,model->allnz,model->allnx);        
                normalize_all(hessian_grad_pp,hessian_grad_ps,model->allnz,model->allnx);                                
            }


            char imagepath_pp[1024];
            sprintf(imagepath_pp, "%s/hessian_delta_pp_%d_%d_%d_%d.dat",model->imagedir,iter,model->allnx,model->allnz,max_inner);
            FILE *fpp = NULL;
            fpp = fopen(imagepath_pp,"wb");
            fwrite(hessian_delta_pp,model->allnx*model->allnz*sizeof(float),1,fpp);
            fclose(fpp);

            sprintf(imagepath_pp, "%s/hessian_delta_ps_%d_%d_%d_%d.dat",model->imagedir,iter,model->allnx,model->allnz,max_inner);
            fpp = fopen(imagepath_pp,"wb");
            fwrite(hessian_delta_ps,model->allnx*model->allnz*sizeof(float),1,fpp);
            fclose(fpp);    


            MPI_Bcast(hessian_delta_pp, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);
            MPI_Bcast(hessian_delta_ps, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);  
            MPI_Bcast(hessian_grad_pp, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);
            MPI_Bcast(hessian_grad_ps, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);                       
            MPI_Barrier(groupcomm);
        }


//calculate alpha &  update m
        MPI_Barrier(groupcomm);
        fenzi =0;
        fenmu =0;

        misfit_step0 = 0;
        misfit_step1 = 0;
        misfit_step2 = 0;

        misfit_step0_ot = 0;
        misfit_step1_ot = 0;
        misfit_step2_ot = 0;

		MPI_Send(send,9,MPI_INT,0,0,MPI_COMM_WORLD);
		for(;;)
		{
			MPI_Recv(recv,9,MPI_INT,0,99,MPI_COMM_WORLD,&status);

            int sno = recv[0];
            int ntr = recv[1];
            int scx = recv[2];
            int scy = recv[3];
            int cx_min_s = recv[4];
            int cy_min_s = recv[5];
            int cx_max_s = recv[6];
            int cy_max_s = recv[7];
            int pos = recv[8];

			if(sno == 0)
			{
				// printf("myid=%d,update image finished,waiting...\n",myid);	
				break;
			}

            if( sno<model->minshot||sno>model->maxshot ) {
                // printf(" the %dth shot is out of imaging range\n", sno);
                // fflush(stdout);
			    MPI_Send(send,9,MPI_INT,0,myid,MPI_COMM_WORLD); 
                continue;               
            }

			printf("Update image, %d Shot in Processor %d\n",sno,myid);

		    int   ngx_left  = cx_min_s;
			int   ngx_right = cx_max_s;
			int   nx_s = (cx_max_s - cx_min_s)/model->dx + 1 ;
            // printf("all_left=%d,ngx_left=%d,ngx_right=%d\n nx_s=%d,pos=%d,ntr=%d\n",model->all_left,ngx_left,ngx_right,nx_s,pos,ntr);
	        int delta_left =int((ngx_left-model->all_left)/model->dx);

            if(ntr==image2delastic.ntr&&ntr==model->ntr_pre){
                read_shot_gather_su(model->fn1, pos, ntr, model->nt, model->record_z, model->gc);
                read_shot_gather_su(model->fn2, pos, ntr, model->nt, model->record_x, model->gc);         
                image2delastic.record_copytoGPU(model->record_z,model->record_x,model->gc);
            }
            else{
                printf("Trace numbers differ from preload ntr_pre in this shot. Need reallocate.\n");
		        image2delastic.ntr = ntr;
                delete[] model->record_z;
                delete[] model->record_x;
                delete[] model->gc;                
                model->record_z = new float[model->nt*ntr]{};
                model->record_x = new float[model->nt*ntr]{};                
                model->gc = new int[ntr]{};
                read_shot_gather_su2(model->fn1, pos, ntr, model->nt, model->record_z, model->gc);  
                read_shot_gather_su2(model->fn2, pos, ntr, model->nt, model->record_x, model->gc); 		               		
                image2delastic.record_copytoGPU(model->record_z,model->record_x,model->gc);
		        std::cout<<"reallocate completed"<<std::endl;
            }


            if(nx_s==model->nx){
			    image2delastic.bufferVpVsHtoD(delta_left+record_left_in_v);
            }
            else{
                printf("Nx differ from preload ntr_pre in this shot. Need reallocate.\n");
                //TODO reallocate space and memory copy
			    image2delastic.bufferVpVsHtoD(delta_left+record_left_in_v);
            }
      
            image2delastic.delta_left = delta_left;
            image2delastic.record_left_in_v = record_left_in_v;            
	        image2delastic.isx = model->pml+(int)((scx-ngx_left)/model->dx);
            image2delastic.isz = model->pml + sy;
            image2delastic.igz = model->pml + gy;
            image2delastic.cmin = delta_left+record_left_in_v;          //most left position(grid) in velocity/image model 
            image2delastic.ngx_left = ngx_left;                         //most left positon   
			int Dim = model->nxpml*model->nzpml;
            // printf("scx = %d,isx = %d\n",scx,image2delastic.isx);
			// printf("source num=%d\n",sno);
		    // printf("gc1=%d,gc2=%d\n",model->gc[0],model->gc[ntr-1]);     

////////////////////////

            CALCULATE_ALPHA_parabola_mb(image2delastic,sno,Dim,myid,model->record_z,model->record_x,model->gc,model->rbc,model->cpu_mem,model->iointerval,iter,image_pp,image_ps,hessian_grad_pp,hessian_grad_ps,\
            misfit_step0,misfit_step1,misfit_step2,step1,step2,&misfit_step0_ot,&misfit_step1_ot,&misfit_step2_ot);                

//calculating instead of memory

			MPI_Send(send,9,MPI_INT,0,myid,MPI_COMM_WORLD);
		}
		printf("myid=%d,update image finished,waiting...\n",myid);	
        MPI_Barrier(groupcomm);
        if(myid==1){
            MPI_Reduce(MPI_IN_PLACE, &fenzi, 1, MPI_DOUBLE, MPI_SUM, 0, groupcomm); 
            MPI_Reduce(MPI_IN_PLACE, &fenmu, 1, MPI_DOUBLE, MPI_SUM, 0, groupcomm);       
            MPI_Reduce(MPI_IN_PLACE, &misfit_step0, 1, MPI_DOUBLE, MPI_SUM, 0, groupcomm); 
            MPI_Reduce(MPI_IN_PLACE, &misfit_step1, 1, MPI_DOUBLE, MPI_SUM, 0, groupcomm); 
            MPI_Reduce(MPI_IN_PLACE, &misfit_step2, 1, MPI_DOUBLE, MPI_SUM, 0, groupcomm);   

            MPI_Reduce(MPI_IN_PLACE, &misfit_step0_ot, 1, MPI_FLOAT, MPI_SUM, 0, groupcomm); 
            MPI_Reduce(MPI_IN_PLACE, &misfit_step1_ot, 1, MPI_FLOAT, MPI_SUM, 0, groupcomm); 
            MPI_Reduce(MPI_IN_PLACE, &misfit_step2_ot, 1, MPI_FLOAT, MPI_SUM, 0, groupcomm);                                    
        }
        else{
            MPI_Reduce(&fenzi, &fenzi, 1, MPI_DOUBLE, MPI_SUM, 0, groupcomm); 
            MPI_Reduce(&fenmu, &fenmu, 1, MPI_DOUBLE, MPI_SUM, 0, groupcomm);    
            MPI_Reduce(&misfit_step0, &misfit_step0, 1, MPI_DOUBLE, MPI_SUM, 0, groupcomm);                
            MPI_Reduce(&misfit_step1, &misfit_step1, 1, MPI_DOUBLE, MPI_SUM, 0, groupcomm);           
            MPI_Reduce(&misfit_step2, &misfit_step2, 1, MPI_DOUBLE, MPI_SUM, 0, groupcomm);      

            MPI_Reduce(&misfit_step0_ot, &misfit_step0_ot, 1, MPI_FLOAT, MPI_SUM, 0, groupcomm);                
            MPI_Reduce(&misfit_step1_ot, &misfit_step1_ot, 1, MPI_FLOAT, MPI_SUM, 0, groupcomm);           
            MPI_Reduce(&misfit_step2_ot, &misfit_step2_ot, 1, MPI_FLOAT, MPI_SUM, 0, groupcomm);                                   
        }

		if(myid==1){

            std::cout<<"sum of fenzi = "<<fenzi<<std::endl;
            std::cout<<"sum of fenmu = "<<fenmu<<std::endl;

            std::cout<<"sum of misfit_step0 = "<<misfit_step0<<std::endl;
            std::cout<<"sum of misfit_step1 = "<<misfit_step1<<std::endl;
            std::cout<<"sum of misfit_step2 = "<<misfit_step2<<std::endl;

            if(update_mp){
                cal_step2(misfit_step0,misfit_step1,misfit_step2,step1,step2,alpha);            

                // cal_step2(misfit_step0_ot,misfit_step1_ot,misfit_step2_ot,step1,step2,alpha);                                
            }

            std::cout<<"iter = "<<iter<<"\talpha = "<<alpha<<std::endl;

            if(update_mb||update_mp){
                if(alpha>300)
                {
                    alpha = 300;
                }
                if(alpha<-100)
                {
                    alpha = -100;
                }
            }


            char misfitdir[1024];
            sprintf(misfitdir, "%s/misfit.txt",model->imagedir);            
            FILE *fpout = fopen(misfitdir,"a");
            fprintf(fpout,"i = %d,alpha = %5.3e\n",iter,alpha);
            fclose(fpout);


            #pragma omp parallel for num_threads(16)
            for(int i=0;i<model->allnx;i++)
                for(int j=0;j<model->allnz;j++)
                {
                    *((model->velp) + i * model->allnz + j) = *((model->velp) + i * model->allnz + j) - alpha*hessian_grad_pp[i*model->allnz+j];
                    *((model->vels) + i * model->allnz + j) = *((model->vels) + i * model->allnz + j) - alpha*hessian_grad_ps[i*model->allnz+j];                          
                }    


            for(int i=0;i<model->allnx;i++)
                for(int j=0;j<model->allnz;j++)
                {
                    if(j<cut_layer){
                        *((model->velp) + i * model->allnz + j) = *((model->velp) + 0 * model->allnz + 0) ;
                        *((model->vels) + i * model->allnz + j) = *((model->vels) + 0 * model->allnz + 0) ;               
                    }          
                }  


            char imagepath_pp[1024];
            FILE *fpp = NULL;

            sprintf(imagepath_pp, "%s/vel_p_%d_%d_%d.dat",model->imagedir,iter,model->allnx,model->allnz);
            fpp = fopen(imagepath_pp,"wb");
            fwrite(model->velp,model->allnx*model->allnz*sizeof(float),1,fpp);
            fclose(fpp); 

            sprintf(imagepath_pp, "%s/vel_s_%d_%d_%d.dat",model->imagedir,iter,model->allnx,model->allnz);
            fpp = fopen(imagepath_pp,"wb");
            fwrite(model->vels,model->allnx*model->allnz*sizeof(float),1,fpp);
            fclose(fpp);   

        }

	    MPI_Bcast(image_pp, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);
	    MPI_Bcast(image_ps, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);

	    MPI_Bcast(model->velp, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);
	    MPI_Bcast(model->vels, model->allnx*model->allnz, MPI_FLOAT, 0, groupcomm);

	    MPI_Barrier(MPI_COMM_WORLD);
	}

    }

    delete[] pp_cg;
    delete[] ps_cg;
    delete[] pp_cg_old;
    delete[] ps_cg_old;    

    delete[]b_pp_grad;
    delete[]b_ps_grad;
    delete[]b_pp_grad2;
    delete[]b_ps_grad2;    

    delete[]sum_pp_grad;
    delete[]sum_ps_grad;    

    
    delete[]hessian_p_pp;
    delete[]hessian_p_ps;
    delete[]hessian_Hp_pp;
    delete[]hessian_Hp_ps;
    delete[]hessian_r_pp;
    delete[]hessian_r_ps;
    delete[]hessian_delta_pp;
    delete[]hessian_delta_ps;    
    delete[]hessian_grad_pp;
    delete[]hessian_grad_ps;        

}


////  main function /////
int main(int argc,char *argv[]){

	char parfn[1024];
    int i,j,isx,isz;
	strcpy(parfn,argv[1]);
	modelpar model;
	char fvelp[1024];
	char fvels[1024];
    char frho[1024];          
 
	float dx,dz;
	int pml,allnx,allnz,nx,nz,dis_shot,scale,order;
	float t;
	int mode;
    int nop;
    int minshot,maxshot;
    int sy,gy;
	float f0;
	int maxiter,iter_round1;
	bool light,rbc,cpu_mem;
	int light_temp,rbc_temp,cpumem_temp,iointerval;
    scale =1 ;     
	FILE *fp = NULL;
	fp = fopen(parfn,"r");
	fscanf(fp,"%f %f %d %d %d %d %d %d %d",&dx,&dz,&pml,&allnx,&allnz,&minshot,&maxshot);       
	fscanf(fp,"%s",fvelp);
	fscanf(fp,"%s",fvels);    
	fscanf(fp,"%s",frho);       
	fscanf(fp,"%s",model.fn1);
	fscanf(fp,"%s",model.fn2);
	fscanf(fp,"%s",model.imagedir);     
	fscanf(fp,"%f %d %d",&f0,&maxiter,&iter_round1);
	fscanf(fp,"%d %d",&sy,&gy);    
	fscanf(fp,"%d %d %d %d",&light_temp,&rbc_temp,&cpumem_temp,&iointerval);  	   	
	fclose(fp);

	int myid,np;
	MPI_Status status;
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Comm_size(MPI_COMM_WORLD,&np);

	light = light_temp;
	rbc = rbc_temp;
	cpu_mem = cpumem_temp;
	
	if(myid==0){
        if(access(model.imagedir, F_OK) == 0){
            std::cout<<"image output directory exist."<<std::endl;
        }
        else{
            mkdir(model.imagedir,0755);
            std::cout<<"mkdir image directory."<<std::endl;
        }      
		printf("minshot=%d,maxshot=%d\n",minshot,maxshot);
	}

	float *image_pp = new float[allnx*allnz]{};
	float *image_ps = new float[allnx*allnz]{};

	float *image_pp_m = new float[allnx*allnz]{};
	float *image_ps_m = new float[allnx*allnz]{};

	float *pp_grad = new float[allnx*allnz]{};
	float *ps_grad = new float[allnx*allnz]{};

    float *illumination;
    illumination = new float[allnx*allnz]{};

	float idz = 1.0f/dz;
	float idx = 1.0f/dx;

	int nt,ns;
	float dt;
	float x0 = 0.0f;
    float cmin,cmax,cmleft,cmright;
	cmin = x0;
    cmax = x0 + (allnx-1)*dx; 

    int **table= new int*[10000] ;   //
    for (i = 0; i < 10000; i++) {
        table[i] = new int[9];
    }

    int all_left = 999999999;

    if(myid ==0){
        if(index_shot(model.fn1, &nt, &dt, &ns, table)){
             printf("Can not read the shot file!\n");
             return 0;
        }

        for(i=0;i<ns;i++){
        //	std::cout<<table[i][4]<<std::endl;
            all_left = MIN(all_left,table[i][4]);
        }        
        //edges of Imaging
        cmleft  = 999999;
        cmright = -999999;
        for (i=0;i<ns;i++){
            if(cmleft>table[i][4])cmleft=table[i][4];
            if(cmright<table[i][6])cmright=table[i][6];
            if(cmin>table[i][4])printf(" Warning! The %dth Shot's minimum coordinate is on the left of velocity model\n",i);
            if(cmax<table[i][6])printf(" Warning! The %dth Shot's maximum coordinate is on the right of velocity model\n",i);
        }
    }
    if(myid==0){
          printf("==========Parameters of input seismic file============\n");
          printf(" Shot number           : %d\n",ns);
          printf(" Sampling point number : %d\n",nt);
          printf(" Sampling interval     : %f s\n",dt);
          printf("======================================================\n\n");
        //   for(i=0;i<ns;i++)printf(" table[%d][8] : %d\n",i,table[i][8]);
    }
    int ntr_pre;
    if(myid==0){
        ntr_pre = table[0][1];
        nx = (int)(table[0][6]/dx+0.5) - (int)(table[0][4]/dx+0.5) + 1;
        nz = allnz;
        printf("nx=%d,nz=%d\n",nx,nz);     
    }
    MPI_Bcast(&all_left, 1, MPI_INT, 0, MPI_COMM_WORLD);    
    MPI_Bcast(&ns, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ntr_pre, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nx, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nz, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);    
	MPI_Barrier(MPI_COMM_WORLD);  
	int nzpml = nz + 2*pml;
	int nxpml = nx + 2*pml;

	model.velp = NULL;	
	model.vels = NULL;	
	model.rho = NULL;	    
	model.velp = new float[allnz*allnx];
	model.vels = new float[allnz*allnx];
	model.rho = new float[allnz*allnx];


	FILE *fp1 = NULL;
    FILE *fp2 = NULL;
    FILE *fp3 = NULL;
	fp1 = fopen(fvelp,"rb");
	fp2 = fopen(fvels,"rb");
	fp3 = fopen(frho,"rb");
	for(i=0; i<allnx; i++)
	{
		fread(&model.velp[i*allnz],sizeof(float),allnz,fp1);
		fread(&model.vels[i*allnz],sizeof(float),allnz,fp2);        
		fread(&model.rho[i*allnz],sizeof(float),allnz,fp3);              
	}
	fclose(fp1);
	fclose(fp2);
	fclose(fp3);

	for(i=0; i<allnx*allnz; i++)
	{
        model.rho[i] = 2000;          
	}

	model.sou = NULL;
	model.sou = new float[nt];

	ricker_wave(model.sou,nt,dt,f0);

    model.record_z = new float[nt*ntr_pre];
    model.record_x = new float[nt*ntr_pre];
	memset(model.record_z,0,sizeof(float)*nt*ntr_pre);
	memset(model.record_x,0,sizeof(float)*nt*ntr_pre);

    model.gc = new int[ntr_pre];

    nop = 4;

    int coordinate_scale = 1;
    int record_left_in_v = 0;    
	MPI_Barrier(MPI_COMM_WORLD);
	init_modelparameters(&model,dx,dz,dt,minshot,maxshot,nx,nz,ns,nxpml,nzpml,allnx,allnz,scale,pml,nt,nop,ntr_pre,light,rbc,cpu_mem,iointerval,all_left);
    all_iteration(myid,np,sy,gy,status,table,&model,image_pp,image_ps,pp_grad,ps_grad,illumination,maxiter,iter_round1,record_left_in_v);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	delete[] model.velp;
	delete[] model.vels;    
	delete[] model.rho;      
	delete[] model.sou;

    for (i = 0; i < 10000; i++){
        delete[] table[i];
    }
    delete[] table;

    delete[] model.record_z;
    delete[] model.record_x; 

    delete[] model.gc;
    delete[] image_pp;
    delete[] image_ps;  
    delete[] image_pp_m;
    delete[] image_ps_m;          
    delete[] pp_grad;
    delete[] ps_grad;
	return 0;

}











