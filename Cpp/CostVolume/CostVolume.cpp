// Free for non-commercial, non-military, and non-critical 
// use unless incorporated in OpenCV. 
// Inherits OpenCV License if in OpenCV.

#include "CostVolume.hpp"
#include "CostVolume.cuh"

#include <opencv2/core/operations.hpp>
#include <opencv2/gpu/stream_accessor.hpp>
#include <opencv2/gpu/device/common.hpp>

#include "utils/utils.hpp"
#include "utils/tinyMat.hpp"
#include "graphics.hpp"
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::gpu;

void CostVolume::solveProjection(const cv::Mat& R, const cv::Mat& T) {
    Mat P;
    RTToP(R, T, P);
    projection.create(4, 4, CV_64FC1);
    projection=0.0;
    projection(Range(0, 2), Range(0, 3)) += cameraMatrix.rowRange(0, 2);

    projection.at<double>(2,3)=1.0;
    projection.at<double>(3,2)=1.0;
    projection=projection*P;
}

void CostVolume::checkInputs(const cv::Mat& R, const cv::Mat& T,
        const cv::Mat& _cameraMatrix) {
    assert(R.size() == Size(3, 3));
    assert(R.type() == CV_64FC1);
    assert(T.size() == Size(1, 3));
    assert(T.type() == CV_64FC1);
    assert(_cameraMatrix.size() == Size(3, 3));
    assert(_cameraMatrix.type() == CV_64FC1);
}

#define FLATUP(src,dst) {GpuMat tmp;tmp.upload(src);dst.create(1,rows*cols, src.type());dst=dst.reshape(0,rows);}

#define FLATALLOC(n) n.create(1,rows*cols, CV_32FC1);n=n.reshape(0,rows)

CostVolume::CostVolume(Mat image, FrameID _fid, int _layers, float _near,
                       float _far, cv::Mat R, cv::Mat T, cv::Mat _cameraMatrix,
                       float initialCost, float initialWeight):
  R(R), T(T), initialWeight(initialWeight), _cuArray(0)
{
    //For performance reasons, OpenDTAM only supports multiple of 32 image sizes with cols >= 64
    CV_Assert(image.rows % 32 == 0 && image.cols % 32 == 0 && image.cols >= 64);
    CV_Assert(_layers>=8);
    
    checkInputs(R, T, _cameraMatrix);
    fid           = _fid;
    rows          = image.rows;
    cols          = image.cols;
    layers        = _layers;
    near          = _near;
    far           = _far;
    depthStep     = (near - far) / (layers - 1);
    cameraMatrix  = _cameraMatrix.clone();
    solveProjection(R, T);
    FLATALLOC(lo);
    FLATALLOC(hi);
    FLATALLOC(loInd);
    dataContainer.create(layers, rows * cols, CV_32FC1);
    
    GpuMat tmp;
    baseImage.upload(image.reshape(0,1));
    cvtColor(baseImage,baseImageGray,CV_RGB2GRAY);
    baseImage=baseImage.reshape(0,rows);
    baseImageGray=baseImageGray.reshape(0,rows);
    cvStream.enqueueMemSet(loInd,0.0);
    cvStream.enqueueMemSet(dataContainer,initialCost);
    data = (float*) dataContainer.data;
    hits = (float*) hitContainer.data;
    count = 0;
    
    //messy way to disguise cuda objects
    _cuArray=Ptr<char>((char*)(new cudaArray*));
    *((cudaArray**)(char*)_cuArray)=0;
    _texObj=Ptr<char>((char*)(new cudaTextureObject_t));
    *((cudaTextureObject_t*)(char*)_texObj)=0;
}

void CostVolume::simpleTex(const Mat& image,Stream cvStream)
{
    cudaArray*& cuArray=*((cudaArray**)(char*)_cuArray);
    cudaTextureObject_t& texObj=*((cudaTextureObject_t*)(char*)_texObj);
    assert(image.isContinuous());
    assert(image.type()==CV_8UC4);
    
    //Describe texture
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModeLinear;
    texDesc.readMode         = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 0;
    cudaChannelFormatDesc channelDesc = //{8, 8, 8, 8, cudaChannelFormatKindUnsigned};
    cudaCreateChannelDesc<uchar4>();
    //Fill Memory
    if (!cuArray) {
      cudaSafeCall(cudaMallocArray(&cuArray, &channelDesc, image.cols, image.rows));
    }
    
    assert((image.dataend-image.datastart)==image.cols*image.rows*sizeof(uchar4));
    
    cudaSafeCall(cudaMemcpyToArrayAsync(cuArray, 0, 0, image.datastart, image.dataend-image.datastart,
                                   cudaMemcpyHostToDevice,StreamAccessor::getStream(cvStream)));
    
    // Specify texture memory location
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    
    if (!texObj) {
      // Create texture object
      cudaSafeCall(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    }
   //return texObj;
}

void CostVolume::preprocessImage(const Mat& inImage, Mat& outImage)
{
    outImage = inImage; //copy only header
    if(inImage.type()!=CV_8UC4 || !inImage.isContinuous()) {
        if(!inImage.isContinuous() && inImage.type()==CV_8UC4) {
            cBuffer.create(inImage.rows,inImage.cols,CV_8UC4);
            outImage=cBuffer;//.createMatHeader();
            inImage.copyTo(outImage);//copies data
        }
        if(inImage.type()!=CV_8UC4) {
            cBuffer.create(inImage.rows,inImage.cols,CV_8UC4);
            Mat cm=cBuffer;//.createMatHeader();
            if(inImage.type()==CV_8UC1||inImage.type()==CV_8SC1) {
                cvtColor(inImage,cm,CV_GRAY2BGRA);
            }else if(inImage.type()==CV_8UC3||inImage.type()==CV_8SC3) {
                cvtColor(inImage,cm,CV_BGR2BGRA);
            }else{
                outImage=inImage;
                if(inImage.channels()==1) {
                    cvtColor(outImage,outImage,CV_GRAY2BGRA);
                }
                if(inImage.channels()==3) {
                    cvtColor(outImage,outImage,CV_BGR2BGRA);
                }
                //outImage is now 4 channel, unknown depth but not 8 bit
                if(inImage.depth()>=5) {//float
                    outImage.convertTo(cm,CV_8UC4,255.0);
                }else if(outImage.depth()>=2) {//0-65535
                    outImage.convertTo(cm,CV_8UC4,1/256.0);
                }
            }
            outImage=cm;
        }
    }
    CV_Assert(outImage.type()==CV_8UC4);
}

void CostVolume::updateCost(const Mat& _image, const cv::Mat& R, const cv::Mat& T)
{
    using namespace cv::gpu::device::dtam_updateCost;
    localStream = cv::gpu::StreamAccessor::getStream(cvStream);
    assert(localStream);
    assert(baseImage.isContinuous() && lo.isContinuous() && hi.isContinuous() && loInd.isContinuous());
    
    Mat image;
    preprocessImage(_image, image);
        
    //change input image to a texture
    simpleTex(image,cvStream);
    cudaTextureObject_t& texObj=*((cudaTextureObject_t*)(char*)_texObj);

    //find projection matrix from cost volume to image (3x4)
    Mat viewMatrixImage;
    RTToP(R,T,viewMatrixImage);

    Mat cameraMatrixTex(3,4,CV_64FC1);
    cameraMatrixTex=0.0;
    cameraMatrix.copyTo(cameraMatrixTex(Range(0,3),Range(0,3)));
    cameraMatrixTex(Range(0,2), Range(2,3)) += 0.5; // add 0.5 to x, y out. removing causes crash!

    Mat imFromWorld=cameraMatrixTex*viewMatrixImage; // 3x4
    Mat imFromCV=imFromWorld*projection.inv();
    imFromCV.colRange(2,3)*=depthStep;
    
    //for each slice
    double *p = (double*)imFromCV.data;
    m34 persp;
    for(int i=0;i<12;i++)
      persp.data[i]=p[i];

    float w=count++ + initialWeight;
    w/=(w+1); 

    globalWeightedBoundsCostCaller(persp, w, rows, cols, layers, rows*cols, hits, data,
                                   (float*)(lo.data), (float*)(hi.data), (float*)(loInd.data),
                                   (float3*)(baseImage.data), (float*)baseImage.data,
                                   texObj);
}

CostVolume::~CostVolume() {
    cudaArray*& cuArray=*((cudaArray**)(char*)_cuArray);
    cudaTextureObject_t& texObj=*((cudaTextureObject_t*)(char*)_texObj);
    if (cuArray) {
        cudaFreeArray(cuArray);
    }
    if (texObj) {
        cudaDestroyTextureObject(texObj);
    }
}
