// Free for non-commercial, non-military, and non-critical
// use unless incorporated in OpenCV.
// Inherits OpenCV Licence if in OpenCV.

//This file does Q and D optimization steps on the GPU
#include "DepthmapDenoiseWeightedHuber.hpp"
#include "DepthmapDenoiseWeightedHuber.cuh"
#include <opencv2/gpu/stream_accessor.hpp>

namespace cv{
    namespace gpu{
using namespace std;
using namespace cv::gpu;

DepthmapDenoiseWeightedHuberImpl::DepthmapDenoiseWeightedHuberImpl(const GpuMat& _visibleLightImage,
                                                        Stream _cvStream) : 
                                                        visibleLightImage(_visibleLightImage), 
                                                        rows(_visibleLightImage.rows),
                                                        cols(_visibleLightImage.cols),
                                                        cvStream(_cvStream)
{
    alloced=false;
    cachedG=false; 
    dInited=false;
}

Ptr<DepthmapDenoiseWeightedHuber>
CV_EXPORTS createDepthmapDenoiseWeightedHuber(InputArray visibleLightImage, Stream cvStream){
    return Ptr<DepthmapDenoiseWeightedHuber>(new DepthmapDenoiseWeightedHuberImpl(visibleLightImage.getGpuMat(),cvStream));
}

// allocate buffer b such that it is continous in memory and has the shape of a
#define CONTINUOUS_ALLOC(b, a) b.create(1, a.rows*a.cols, CV_32FC1); b=b.reshape(0, a.rows);

void DepthmapDenoiseWeightedHuberImpl::allocate(int rows, int cols)
{
    if(!(rows % 32 == 0 && cols % 32 == 0 && cols >= 64)){
        CV_Assert(!"For performance reasons, DepthmapDenoiseWeightedHuber currently supports ");
        CV_Assert(!"only multiples of 32 image sizes with cols >= 64. Pad the image to achieve this.");
    }

    // organize this better. repeated code in operator()
    if(!_a.data){
        _a.create(1,rows*cols, CV_32FC1);
        _a=_a.reshape(0, rows);

        _q.create(1, 2*rows*cols, CV_32FC1);
        _q=_q.reshape(0, 2*rows);
    }
    
    CONTINUOUS_ALLOC(_d, _a);
    CONTINUOUS_ALLOC(_g, _a);

    alloced=true;
}

void DepthmapDenoiseWeightedHuberImpl::computeSigmas(float epsilon, float theta)
{
    /*
    //This function is my best guess of what was meant by the line:
    //"Gradient ascent/descent time-steps sigma_q , sigma_d are set optimally
    //for the update scheme provided as detailed in [3]."
    // Where [3] is :
    //A. Chambolle and T. Pock. A first-order primal-dual 
    //algorithm for convex problems with applications to imaging.
    //Journal of Mathematical Imaging and Vision, 40(1):120-
    //145, 2011. 3, 4, 6
    //
    // I converted these mechanically to the best of my ability, but no 
    // explaination is given in [3] as to how they came up with these, just 
    // some proofs beyond my ability.
    //
    // Explainations below are speculation, but I think good ones:
    //
    // L appears to be a bound on the largest vector length that can be 
    // produced by the linear operator from a unit vector. In this case the 
    // linear operator is the differentiation matrix with G weighting 
    // (written AG in the DTAM paper,(but I use GA because we want to weight 
    // the springs rather than the pixels)). Since G has each row sum < 1 and 
    // A is a forward difference matrix (which has each input affecting at most
    // 2 outputs via pairs of +-1 weights) the value is bounded by 4.0.
    //
    // So in a sense, L is an upper bound on the magnification of our step size.
    // 
    // Lambda and alpha are called convexity parameters. They come from the 
    // Huber norm and the (d-a)^2 terms. The convexity parameter of a function 
    // is defined as follows: 
    //  Choose a point on the function and construct a parabola of convexity 
    //    c tangent at that point. Call the point c-convex if the parabola is 
    //    above the function at all other points. 
    //  The smallest c such that the function is c-convex everywhere is the 
    //      convexity parameter.
    //  We can think of this as a measure of the bluntest tip that can trace the 
    //     entire function.
    // This is important because any gradient descent step that would not 
    // cause divergence on the tangent parabola is guaranteed not to diverge 
    // on the base function (since the parabola is always higher(i.e. worse)).
    */
  
    //lower is better(longer steps), but in theory only >=4 is guaranteed to converge. For the adventurous, set to 2 or 1.44        
    float L=4;
    
    float lambda = 1.0/theta;
    float alpha  = epsilon;
    
    float mu = 2.0*sqrt(lambda*alpha)/L;

    sigma_d = mu/(2.0*lambda);
    sigma_q = mu/(2.0*alpha);
}

void DepthmapDenoiseWeightedHuberImpl::cacheGValues(InputArray visibleLightImage){
    using namespace cv::gpu::device::dtam_denoise;

    localStream = cv::gpu::StreamAccessor::getStream(cvStream);

    // TODO this should only be done once, why recheck in operator()
    if(!alloced)
        allocate(rows,cols);

    if(!visibleLightImage.empty()) {
        img = visibleLightImage.getGpuMat();
        cachedG=false;
    }
    if(cachedG)
        return;
    
    // Call the gpu function for caching g's
    computeGCaller((float*)img.data, (float*)_g.data, cols, rows, img.step);

    cachedG=true;
}

GpuMat DepthmapDenoiseWeightedHuberImpl::operator()(InputArray _ain, float epsilon, float theta){
    const GpuMat& ain=_ain.getGpuMat();
    
    using namespace cv::gpu::device::dtam_denoise;
    
    rows=ain.rows;
    cols=ain.cols;
    
    if(ain.empty || !ain.isContinuous()){
        _a.create(1,rows*cols, CV_32FC1);
        _a=_a.reshape(0,rows);
        cvStream.enqueueCopy(ain,_a);
    }
    else {
        _a=ain;
    }
    
    if(!alloced){
        allocate(rows,cols);
    } 
    
    // TODO fix the visibleLightImage snafu
    if(!visibleLightImage.empty())
        cacheGValues();
    if(!cachedG){
        _g=1.0f;
    }
    if(!dInited){
        cvStream.enqueueCopy(_a,_d);
        // TODO is the right place to do this?
        _q = 0.0f;
        dInited=true;
    }
    
    computeSigmas(epsilon,theta);
    
    update_q_dCaller((float*)_g.data, (float*)_a.data,  // const input
                     (float*)_q.data, (float*)_d.data,  // input q, d
                     cols, rows, // dimensions
                     sigma_q, sigma_d, epsilon, theta // parameters
                     );
    cudaSafeCall(cudaGetLastError());
    return _d;
}
}  
}


