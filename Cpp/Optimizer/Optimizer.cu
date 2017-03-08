#include <assert.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "Optimizer.cuh"

#ifndef __CUDACC__
#define __constant__
#define __global__
#define __host__
#define __device__
#endif

namespace cv { namespace gpu { namespace device {
    namespace dtam_optimizer{

static unsigned int arows,acols;

#define SEND(type,sym) cudaMemcpyToSymbol(sym, &h_ ## sym, sizeof(type));

void loadConstants(uint h_rows, uint h_cols, uint, uint,
        float*, float* , float* , float* , float* ,
        float*) {

        //special
        arows=h_rows;
        acols=h_cols;
        assert(arows>0);
}

cudaStream_t localStream=0;

const int BLOCKX1D=256;

#define GENERATE_CUDA_FUNC1D(funcName,arglist,notypes)      \
    static __global__ void funcName arglist;                \
    void funcName##Caller arglist{                          \
      dim3 dimBlock(BLOCKX1D);                              \
      dim3 dimGrid((acols*arows) / dimBlock.x);             \
      funcName<<<dimGrid, dimBlock,0,localStream>>>notypes; \
      cudaSafeCall( cudaGetLastError() );                   \
    };static __global__ void funcName arglist


__device__
static inline float afunc(float costval,float theta,float d,float ds,float a,float lambda){
//    return costval;
    return 1.0f/(2.0f*theta)*ds*ds*(d-a)*(d-a) + costval*lambda;//Literal implementation of Eq.14, note the datastep^2 factor to scale correctly
//     return 1.0/(2.0*theta)*(d-a)*(d-a) + data[a]*lambda;//forget the ds^2 factor for better numerical behavior(sometimes)
//     return std::abs(1.0/(2.0*theta)*ds*ds*(d-a)) + data[a]*lambda;//L1 Version
}

//template <int layers>
GENERATE_CUDA_FUNC1D(minimizeA,
                        (float*cdata,float*a, float* d, int layers,float theta,float lambda),
                        (cdata,a,d,layers,theta,lambda)) {
   // __shared__ float s0[32*BLOCKX1D];
   // float* s=s0+threadIdx.x*32;
    unsigned int pt = blockIdx.x * blockDim.x + threadIdx.x;
    float dv=d[pt];
    float *out=a+pt;
    float *cp=cdata+pt;
    const int layerStep=blockDim.x*gridDim.x;
    const int l=layerStep;
    const float depthStep=1.0f/layers;
    float vlast,vnext,v,A,B,C;

    unsigned int mini=0;

    float minv= v = afunc(cp[0], theta ,dv,depthStep,0,lambda);

    vnext = afunc(cp[l], theta ,dv,depthStep,1,lambda);
#pragma unroll 4
    for(unsigned int z=2;z<layers;z++){
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

//    a[pt]=mini;
//    return;//the no interpolation soln

    if (vnext<minv){//last was best
        *out=layers-1;
        return;
    }

    if (mini==0){//first was best
        *out=0;
        return;
    }

    B=minv;//avoid divide by zero, since B is already <= others, make < others

    float denom=(A-2*B+C);
    float delt=(A-C)/(denom*2);
    //value=A/2*(delt)*(delt-1)-B*(delt+1)*(delt-1)+C/2*(delt+1)*(delt);
//    minv=B-(A-C)*delt/4;
    if(denom!=0)
        *out=delt+float(mini);
    else
        *out=mini;//float(mini);
}

GENERATE_CUDA_FUNC1D(minimizeAshared,
                        (float*cdata,float*a, float* d,int rows,int cols, int layers,float theta,float lambda),
                        (cdata,a,d,rows,cols,layers,theta,lambda)) {
    __shared__ float s0[32*BLOCKX1D];
    float* s=s0+threadIdx.x*32;
    unsigned int pt = blockIdx.x * blockDim.x + threadIdx.x;
    float dv=d[pt];
    float *out=a+pt;
    float *cp=cdata+pt;
    int layerStep=rows*cols;
    const float depthStep=1.0f/layers;
    const int l=layerStep;


    float minv=__int_as_float(0x7F800000);//inf
    unsigned int mini=0;
#pragma unroll 8
    for(unsigned int z=0;z<layers;z++){
        float c=cp[z*l];

        float val = afunc(c, theta ,dv,depthStep,z,lambda);
        if(val<minv){
            minv=val;
            mini=z;
        }
        s[z]=val;
    }
//    a[pt]=mini;
//    return;//the no interpolation soln
    if (mini==0){
        *out=0;
        return;
    }
    if (mini==layers-1){
        *out=layers-1;
        return;
    }
    float A=s[mini-1];
    float B=minv;//avoid divide by zero, since B is already <= others, make < others
    float C=s[mini+1];
    float denom=(A-2*B+C);
    float delt=(A-C)/denom*.5;
    //value=A/2*(delt)*(delt-1)-B*(delt+1)*(delt-1)+C/2*(delt+1)*(delt);
//    minv=B-(A-C)*delt/4;
    if(denom!=0)
        *out=delt+float(mini);
    else
        *out=100;//float(mini);
}

//cool but cryptic min algo
GENERATE_CUDA_FUNC1D(minimizeAcool,
                        (float*cdata,float*a, float* d,int rows,int cols, int layers,float theta,float lambda),
                        (cdata,a,d,rows,cols,layers,theta,lambda)) {
    //__shared__ float s[32];
    unsigned int pt = blockIdx.x * blockDim.x + threadIdx.x;
    float dv=d[pt];
    float *cp=cdata+pt;
    int layerStep=rows*cols;
    const float depthStep=1.0f/layers;
    const int l=layerStep;


    float minv=__int_as_float(0x7F800000);//inf
    float mini=0;
#pragma unroll 32
    for(int z=0;z<layers;z++){
        float c=cp[z*l];
        //s[z]=cp[z*l];

        float val = afunc(c, theta ,dv,depthStep,z,lambda);

        val=__int_as_float(__float_as_int(val)&0xFFFFFF00|z);//load the depth into the least significant 8 bits of the float
        if(val<minv){
            minv=val;
            mini=z;
            mini=__float_as_int(minv)&0x000000FF;
        }
//        minv=fminf(minv,val);
    }
    int loc = __float_as_int(minv)&0x000000FF;

    a[pt]=mini;
}

}}}} // namespace
