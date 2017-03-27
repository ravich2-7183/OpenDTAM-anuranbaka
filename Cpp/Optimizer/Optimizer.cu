#include <assert.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "Optimizer.cuh"

namespace cv { namespace gpu { namespace device { namespace dtam_optimizer{

static unsigned int arows, acols;

void loadConstants(uint h_rows, uint h_cols, uint, uint,
                   float*, float*, float*, float*, float*, float*)
{
    arows=h_rows;
    acols=h_cols;
    assert(arows>0);
}

cudaStream_t localStream=0;

const int BLOCKX1D=256;

#define GENERATE_CUDA_FUNC1D(funcName,arglist,notypes)    \
    static __global__ void funcName arglist;              \
    void funcName##Caller arglist{                        \
      dim3 dimBlock(BLOCKX1D);                            \
      dim3 dimGrid((acols*arows) / dimBlock.x);           \
      funcName<<<dimGrid, dimBlock>>>notypes;             \
      cudaSafeCall( cudaGetLastError() );                 \
    };static __global__ void funcName arglist

__device__
static inline float Eaux(float theta, float d, float a, float lambda, float costval)
{
    // TODO is scaling by depthStep = 1.0f/layers required ?
    // removing scaling by depthStep causes oversmoothing, 03/23/2017-19:00, Thursday
    // Eaux1 = 0.5f/theta*depthStep*depthStep*(d[i]-a[i])*(d[i]-a[i]);
    // TODO will i run into the notorious float substraction bug here?
    float ds = 1.0/32.0;
    return 0.5f/theta*(d-a)*(d-a)*ds*ds + lambda*costval;
}
// TODO define layers, theta and lambda as const
GENERATE_CUDA_FUNC1D(minimizeANew,
                     (float*cdata, float*a, float*d, int layers, float theta, float lambda),
                     (      cdata,       a,       d,     layers,       theta,       lambda))
{
    // thread coordinate
    const int   i         = blockIdx.x * blockDim.x + threadIdx.x; // TODO check if i is invalid and return
    const int   layerStep = blockDim.x * gridDim.x;
    const float di        = d[i];

    float minc = Eaux(theta, di, 0, lambda, cdata[i]);
    int   mini = 0;
    #pragma unroll 4 // TODO what does the 4 do?
    for(int z=1; z<layers; z++) {
        float c = Eaux(theta, di, z, lambda, cdata[i+z*layerStep]);
        if(c < minc) {
            minc = c;
            mini = z;
        }
    }

    if(mini == 0 || mini == layers-1) { // first or last was best
        a[i]=float(mini);
        return;
    }
    
    // sublayer sampling as the minimum of the parabola with the 2 points around (mini, minc)
    float A = Eaux(theta, di, mini-1, lambda, cdata[i+(mini-1)*layerStep]);
    float B = minc;
    float C = Eaux(theta, di, mini+1, lambda, cdata[i+(mini+1)*layerStep]);
    float delta = ((A+C)==2*B)? 0.0f : (A-C)/(2*(A-2*B+C));
    a[i] = float(mini) + delta;
}

__device__
static inline float afunc(float costval, float theta, float d, float ds, float a, float lambda)
{
    //Literal implementation of Eq.14, note the depthStep^2 factor to scale correctly
    return 1.0f/(2.0f*theta)*ds*ds*(d-a)*(d-a) + costval*lambda;
}

GENERATE_CUDA_FUNC1D(minimizeAOld,
                        (float*cdata,float*a, float* d, int layers,float theta,float lambda),
                        (cdata,a,d,layers,theta,lambda))
{
    unsigned int pt = blockIdx.x * blockDim.x + threadIdx.x;
    float dv=d[pt];
    float *out=a+pt;
    float *cp=cdata+pt;
    const int layerStep=blockDim.x*gridDim.x;
    const int l=layerStep;
    const float depthStep=1.0f/layers;
    float vlast, vnext, v, A, B, C;

    unsigned int mini=0;
    float minv= v = afunc(cp[0], theta ,dv,depthStep,0,lambda);
    vnext = afunc(cp[l], theta ,dv,depthStep,1,lambda);
    #pragma unroll 4
    for(unsigned int z=2;z<layers;z++) {
        vlast=v;
        v=vnext;
        vnext = afunc(cp[z*l], theta ,dv,depthStep,z,lambda);
        if(v<minv){
            A=vlast;
            C=vnext;
            minv=v;
            mini=z-1;
        }
    }

    if (vnext<minv){//last was best
        *out=layers-1;
        return;
    }

    if (mini==0){//first was best
        *out=0;
        return;
    }

    B=minv;//avoid divide by zero, since B is already <= others, make < others

    float denom = (A-2*B+C);
    float delt  = (A-C) / (2*denom);
    *out = (denom!=0)? (delt+float(mini)) : float(mini);
}

}}}} // namespace
