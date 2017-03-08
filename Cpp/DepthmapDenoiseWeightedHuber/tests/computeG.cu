#include <opencv2/gpu/device/common.hpp> // for cudaSafeCall
#include <opencv2/core/core.hpp> // for CV_Assert
#include "computeG.cuh"

// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// TODO move this to computeG.cuh
__global__ void computeGScharr(float* img, float* g, int width, int height);

void computeGScharrCaller(float* img, float* g, int width, int height, int pitch)
{
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

  // Set texture reference parameters
  texRef.normalized     = false;
  texRef.addressMode[0] = cudaAddressModeClamp;
  texRef.addressMode[1] = cudaAddressModeClamp;
  texRef.filterMode     = cudaFilterModeLinear;

  // Bind the array to the texture reference
  size_t offset;
  cudaBindTexture2D(&offset, texRef, img, channelDesc,
                    width, height, pitch);
 
  // Invoke kernel
  dim3 dimBlock(16, 16);
  dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
               (height + dimBlock.y - 1) / dimBlock.y);
  computeGScharr<<<dimGrid, dimBlock>>>(img, g, width, height);
}

// Scharr gradient kernel
__global__ void computeGScharr(float* img,
                               float* g,
                               int width, int height) 
{
    // Calculate texture coordinates
    float x = (float) (blockIdx.x * blockDim.x + threadIdx.x);
    float y = (float) (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Scharr kernels (combine Gaussian smoothing and differentiation)
    /*  gx           gy
       -3 0  3       -3 -10 -3
      -10 0 10        0   0  0
       -3 0  3        3  10  3
    */

    float gx, gy;
    gx = -3.0f * tex2D(texRef, x-1, y-1) +
          3.0f * tex2D(texRef, x+1, y-1) +
         10.0f * tex2D(texRef, x-1, y  ) +
        -10.0f * tex2D(texRef, x+1, y  ) +
         -3.0f * tex2D(texRef, x-1, y+1) +
          3.0f * tex2D(texRef, x+1, y+1) ;

    gy = -3.0f * tex2D(texRef, x-1, y-1) +
         -3.0f * tex2D(texRef, x+1, y-1) +
        -10.0f * tex2D(texRef, x  , y-1) +
         10.0f * tex2D(texRef, x  , y+1) +
          3.0f * tex2D(texRef, x-1, y+1) +
          3.0f * tex2D(texRef, x+1, y+1) ;
    
    g[(int)(y * width + x)] = sqrtf(gx*gx + gy*gy);
    // g[y * width + x] = expf(alpha*powf(sqrtf(gx*gx + gy*gy), beta));
}

/*  Pseudocode for computeG()
    void computeG(){
    // g0 is the strongest nearby gradient (excluding point defects)
    g0x=fabsf(pr-pl);//|dx|
    g0y=fabsf(pd-pu);//|dy|
    g0=max(g0x,g0y);
    // g1 is the scaled g0 through the g function exp(-alpha*x^beta)
    g1=sqrt(g0); //beta=0.5
    alpha=3.5;
    g1=exp(-alpha*g1);
    //hard to explain this without a picture, but breaks are where both neighboring pixels are near a change
    gx=max(g1r,g1);
    gy=max(g1d,g1);
    gu=gyu;  //upper spring is the lower spring of the pixel above
    gd=gy;   //lower spring
    gr=gx;   //right spring
    gl=gxl;  //left spring is the right spring of the pixel to the left
    }
*/

static __global__ void computeG1(float* pp, float* g1p, float* gxp, float* gyp, int cols);
static __global__ void computeG2(float* pp, float* g1p, float* gxp, float* gyp, int cols);

cudaStream_t localStream=0;

const int BLOCKX2D=32;
const int BLOCKY2D=4;

void computeGCaller(float* pp, float* g1p, float* gxp, float* gyp, int rows, int cols) {
  dim3 dimBlock(BLOCKX2D, BLOCKY2D);
  dim3 dimGrid(1, (rows + dimBlock.y - 1) / dimBlock.y);

  computeG1<<<dimGrid, dimBlock,0,localStream>>>(pp, g1p, gxp, gyp, cols);
  cudaDeviceSynchronize();

  computeG2<<<dimGrid, dimBlock,0,localStream>>>(pp, g1p, gxp, gyp, cols);
  cudaDeviceSynchronize();
   
  cudaSafeCall( cudaGetLastError() );
}

static __global__ void computeG1(float* pp, float* g1p, float* gxp, float* gyp, int cols)
{
#if __CUDA_ARCH__>=300
  //TODO: make compatible with cuda 2.0 and lower (remove shuffles). Probably through texture fetch
  //subscripts u,d,l,r mean up,down,left,right

  const float alpha=3.5f;
  int x = threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int upoff=-(y!=0)*cols;
  int dnoff=(y<gridDim.y*blockDim.y-1)*cols;
  //itr0
  int pt=x+y*cols;
  float ph,pn,pu,pd,pl,pr;
  float g0x,g0y,g0,g1,gt,gsav;
  float tmp;
  ph=pp[pt];
  pn=pp[pt+blockDim.x];

  pr=(threadIdx.x>=30)? __shfl_up(pn,30) : __shfl_down(ph,2);

  pl=ph;
  pu=pp[pt+upoff];
  pd=pp[pt+dnoff];

  // g0 is the strongest nearby gradient (excluding point defects)
  gt=fabsf(pr-pl);
  g0x=__shfl_up(gt,1);//?xxxxxx no prior val
  gsav=__shfl_down(gt,31);//x000000 for next time
  g0x=threadIdx.x>0?g0x:0.0f;//0xxxxxx
  g0y=fabsf(pd-pu);

  g0=fmaxf(g0x,g0y);
  // g1 is the scaled g0 through the g function
  g1=sqrt(g0);
  g1=exp(-alpha*g1);
  //save
  g1p[pt]=g1;

  x+=32;
  //itr 1:n-2
  for(;x<cols-32;x+=32){
    pt=x+y*cols;
    ph=pn;
    pn=pp[pt+blockDim.x];
    pr=__shfl_down(ph,2);
    tmp=__shfl_up(pn,30);
    pr=threadIdx.x>=30?tmp:pr;

    pl=ph;
    pu=pp[pt+upoff];
    pd=pp[pt+dnoff];

    // g0 is the strongest nearby gradient (excluding point defects)
    gt=fabsf(pr-pl);
    g0x=__shfl_up(gt,1);//?xxxxxx
    g0x=threadIdx.x>0?g0x:gsav;//xxxxxxx
    gsav=__shfl_down(gt,31);//x000000 for next time
    g0y=fabsf(pd-pu);

    g0=fmaxf(g0x,g0y);

    // g1 is the scaled g0 through the g function
    g1=sqrt(g0);
    g1=exp(-alpha*g1);
    //save
    g1p[pt]=g1;
  }

  //itr n-1
  pt=x+y*cols;
  ph=pn;
  pr=__shfl_down(ph,2);
  pl=ph;
  pu=pp[pt+upoff];
  pd=pp[pt+dnoff];

  // g0 is the strongest nearby gradient (excluding point defects)
  gt=fabsf(pr-pl);
  g0x=__shfl_up(gt,1);//?xxxxxx
  g0x=threadIdx.x>0?g0x:gsav;//xxxxxxx
  g0y=fabsf(pd-pu);

  g0=fmaxf(g0x,g0y);
  // g1 is the scaled g0 through the g function
  g1=sqrt(g0);
  g1=exp(-alpha*g1);
  //save
  g1p[pt]=g1;
#endif
}

static __global__ void computeG2(float* pp, float* g1p, float* gxp, float* gyp, int cols)
{
#if __CUDA_ARCH__>=300
  int x = threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int dnoff=(y<gridDim.y*blockDim.y-1)*cols;
  //itr0
  int pt=x+y*cols;
  float g1h,g1n,g1u,g1d,g1r,g1l,gx,gy;
  float tmp;
  //part2, find gx,gy
  x = threadIdx.x;
  y = blockIdx.y * blockDim.y + threadIdx.y;
  //itr0
  pt=x+y*cols;

  g1h=g1p[pt];
  g1n=g1p[pt+blockDim.x];
  g1r=__shfl_down(g1h,1);
  tmp=__shfl_up(g1n,31);
  if(threadIdx.x>=31){
    g1r=tmp;
  }
  g1l=g1h;
  g1u=g1h;
  g1d=g1p[pt+dnoff];

  gx=fmaxf(g1l,g1r);
  gy=fmaxf(g1u,g1d);

  //save
  gxp[pt]=gx;
  gyp[pt]=gy;
  x+=32;
  //itr 1:n-2
  for(;x<cols-32;x+=32){
    pt=x+y*cols;
    g1h=g1n;
    g1n=g1p[pt+blockDim.x];
    g1r=__shfl_down(g1h,1);
    tmp=__shfl_up(g1n,31);
    g1r=threadIdx.x>=31?tmp:g1r;

    g1l=g1h;
    g1u=g1h;
    g1d=g1p[pt+dnoff];

    gx=fmaxf(g1l,g1r);
    gy=fmaxf(g1u,g1d);
    //save
    gxp[pt]=gx;
    gyp[pt]=gy;
  }

  //itr n-1
  pt=x+y*cols;
  g1h=g1n;
  g1r=__shfl_down(g1h,1);
  g1l=g1h;
  g1u=g1h;
  g1d=g1p[pt+dnoff];

  gx=fmaxf(g1l,g1r);
  gy=fmaxf(g1u,g1d);


  //save
  gxp[pt]=gx;
  gyp[pt]=gy;
#endif
}

