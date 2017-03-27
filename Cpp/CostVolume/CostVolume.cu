#include <assert.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include "CostVolume.cuh"

namespace cv { namespace gpu { namespace device { namespace dtam_updateCost{

cudaStream_t localStream;

#define CONSTT uint  rows, uint  cols, uint  layers, uint layerStep, float* hdata, float* cdata, float* lo, float* hi, float* loInd, float3* base,  float* bf, cudaTextureObject_t tex
#define CONSTS rows,  cols,  layers, layerStep,  hdata,  cdata,  lo, hi,  loInd,  base,   bf,  tex

__global__ void globalWeightedBoundsCost(m34 p, float weight, CONSTT)
{
    //float*bf=(float*)base;
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float xf=x;
    float yf=y;
    unsigned int offset=x+y*cols;

    float3 B = base[x+y*cols];//Known bug:this requires 12 loads instead of 4 because of stupid memory addressing, can't really fix
    float wi = p.data[8]*xf + p.data[9]*yf + p.data[11];
    float xi = (p.data[0]*xf + p.data[1]*yf + p.data[3]);
    float yi = (p.data[4]*xf + p.data[5]*yf + p.data[7]);
    float minv=1000.0,maxv=0.0;
    float mini=0;
    for(unsigned int z=0;z<layers;z++){
        float c0=cdata[offset+z*layerStep];
        float wiz = wi+p.data[10]*z;
        float xiz = xi+p.data[2] *z;
        float yiz = yi+p.data[6] *z;
        float4 c = tex2D<float4>(tex, xiz/wiz, yiz/wiz);
        float v1 = fabsf(c.x - B.x);
        float v2 = fabsf(c.y - B.y);
        float v3 = fabsf(c.z - B.z);
        float ns=c0*weight+(v1+v2+v3)*(1-weight);
        cdata[offset+z*layerStep]=ns;
        if (ns < minv) {
        minv = ns;
        mini = z;
        }
        maxv=fmaxf(ns,maxv);
    }
    lo[offset]=minv;
    loInd[offset]=mini;
    hi[offset]=maxv;
}

#define BLOCK_X 64
#define BLOCK_Y 4

void globalWeightedBoundsCostCaller(m34 p,float weight,CONSTT){
   dim3 dimBlock(BLOCK_X,BLOCK_Y);
   dim3 dimGrid((cols  + dimBlock.x - 1) / dimBlock.x,
                (rows + dimBlock.y - 1) / dimBlock.y);
   globalWeightedBoundsCost<<<dimGrid, dimBlock, 0, localStream>>>(p, weight,CONSTS);
   assert(localStream);
   cudaSafeCall( cudaGetLastError() );
}

}}}}
