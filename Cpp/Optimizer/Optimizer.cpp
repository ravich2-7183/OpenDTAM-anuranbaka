// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV Licence if in OpenCV.

#include "Optimizer.hpp"
#include "Optimizer.cuh"
#include <opencv2/gpu/stream_accessor.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;
using namespace gpu;
static void memZero(GpuMat& in,Stream& cvStream);

static void memZero(GpuMat& in,Stream& cvStream)
{
    cudaSafeCall(cudaMemsetAsync(in.data,0,in.rows*in.cols*sizeof(float),cv::gpu::StreamAccessor::getStream(cvStream)));
}

Optimizer::Optimizer(CostVolume& cv,
                     float thetaStartIn,
                     float thetaMinIn,
                     float thetaStepIn,
                     float epsilonIn,
                     float lambdaIn) :
thetaStart(thetaStartIn),
thetaMin(thetaMinIn),
thetaStep(thetaStepIn),
epsilon(epsilonIn),
lambda(lambdaIn)
{
    attach(cv);
    initOptimization();
}

void Optimizer::attach(CostVolume& cv)
{
    //For performance reasons, OpenDTAM only supports multiple of 32 image sizes with cols >= 64
    CV_Assert(cv.rows % 32 == 0 && cv.cols % 32 == 0 && cv.cols >= 64);
    allocate();
    stableDepthEnqueued=haveStableDepth=0;
    this->cv=cv;
    cvStream=cv.cvStream;
}

#define FLATALLOC( n) n.create(1,cv.rows*cv.cols, CV_32FC1); n=n.reshape(0,cv.rows);CV_Assert(n.isContinuous())

void Optimizer::allocate()
{}

void Optimizer::initOptimization()
{
    theta=thetaStart;
}

void Optimizer::initA()
{}

bool Optimizer::optimizeA(const cv::gpu::GpuMat _d, cv::gpu::GpuMat _a)
{
    CV_Assert(cv.data||!"Not attached to a CostVolume. Try calling attach() when doing delayed initialization.");
    using namespace cv::gpu::device::dtam_optimizer;

    localStream = cv::gpu::StreamAccessor::getStream(cvStream);

    this->_a=_a;

    loadConstants(cv.rows, cv.cols, cv.layers, cv.rows*cv.cols,
                  (float*)_a.data, (float*)_d.data, cv.data,
                  (float*)cv.lo.data, (float*)cv.hi.data, (float*)cv.loInd.data);
   
    minimizeANewCaller(cv.data, (float*)_a.data, (float*)_d.data, cv.layers, theta, lambda);
    
    theta *= thetaStep;
    
    bool doneOptimizing = (theta <= thetaMin);
    if(doneOptimizing) {
        stableDepthReady=Ptr<char>((char*)(new cudaEvent_t));
        cudaEventCreate((cudaEvent_t*)(char*)stableDepthReady, cudaEventBlockingSync);
        cvStream.enqueueConvert(_a, stableDepth, CV_32FC1, cv.depthStep, cv.far);
        cudaEventRecord(*(cudaEvent_t*)(char*)stableDepthReady, localStream);
        stableDepthEnqueued = 1;
    }
    return doneOptimizing;
}

const cv::Mat Optimizer::depthMap()
{
    //Returns the best available depth map
    // Code should not rely on the particular mapping of true
    // internal data to true inverse depth, as this may change.
    // Currently depth is just a constant multiple of the index, so
    // infinite depth is always represented. This is likely to change.
    
    cv::gpu::Stream str;
    Mat tmp(cv.rows, cv.cols, CV_32FC1);
    if(stableDepthEnqueued) {
        cudaEventSynchronize(*(cudaEvent_t*)(char*)stableDepthReady);
        str.enqueueDownload(stableDepth, tmp);
        str.waitForCompletion();
    }
    else {
        str.enqueueDownload(_a, tmp);
        str.waitForCompletion();
        tmp = tmp * cv.depthStep + cv.far;
    }
    
    return tmp;
}

