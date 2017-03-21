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
//    {//debug
//        cout<<"Augmented Camera Matrix:\n"<<projection<<endl;
//    }
    projection=projection*P;
    
   // exit(0);
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

#define FLATUP(src,dst){GpuMat tmp;tmp.upload(src);dst.create(1,rows*cols, src.type());dst=dst.reshape(0,rows);}
#define FLATALLOC(n) n.create(1,rows*cols, CV_32FC1);n=n.reshape(0,rows)
CostVolume::CostVolume(Mat image, FrameID _fid, int _layers, float _near,
        float _far, cv::Mat R, cv::Mat T, cv::Mat _cameraMatrix,
        float initialCost, float initialWeight): R(R),T(T),initialWeight(initialWeight),_cuArray(0) {

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
    //hitContainer.create(layers, rows * cols, CV_32FC1);
    //hitContainer = initialWeight;
    hits = (float*) hitContainer.data;
    count = 0;
    
    //messy way to disguise cuda objects
    _cuArray=Ptr<char>((char*)(new cudaArray*));
    *((cudaArray**)(char*)_cuArray)=0;
    _texObj=Ptr<char>((char*)(new cudaTextureObject_t));
    *((cudaTextureObject_t*)(char*)_texObj)=0;
    
}

void CostVolume::simpleTex(const Mat& image,Stream cvStream){
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
    if (!cuArray){
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
    
    if (!texObj){
    // Create texture object
    cudaSafeCall(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    }
   //return texObj;
}


void CostVolume::updateCost(const Mat& _image, const cv::Mat& R, const cv::Mat& T){
    using namespace cv::gpu::device::dtam_updateCost;
    localStream = cv::gpu::StreamAccessor::getStream(cvStream);
    
    // 0  1  2  3
    // 4  5  6  7
    // 8  9  10 11
    // 12 13 14 15
    //
    // 0 1 2
    // 3 4 5
    // 6 7 8
    //
    // want cudaReadModeNormalizedFloat for auto convert to [0,1]
    // cudaAddressModeClamp
    // cudaFilterModeLinear
    //
    // make sure we modify the cameraMatrix to take into account the texture coordinates
    //
    Mat image;
    {
    image=_image;//no copy
        if(_image.type()!=CV_8UC4 || !_image.isContinuous()){
            if(!_image.isContinuous()&&_image.type()==CV_8UC4){
                cBuffer.create(_image.rows,_image.cols,CV_8UC4);
                image=cBuffer;//.createMatHeader();
                _image.copyTo(image);//copies data
                
            }
            if(_image.type()!=CV_8UC4){
                cBuffer.create(_image.rows,_image.cols,CV_8UC4);
                Mat cm=cBuffer;//.createMatHeader();
                if(_image.type()==CV_8UC1||_image.type()==CV_8SC1){
                    cvtColor(_image,cm,CV_GRAY2BGRA);
                }else if(_image.type()==CV_8UC3||_image.type()==CV_8SC3){
                    cvtColor(_image,cm,CV_BGR2BGRA);
                }else{
                    image=_image;
                    if(_image.channels()==1){
                        cvtColor(image,image,CV_GRAY2BGRA);
                    }
                    if(_image.channels()==3){
                        cvtColor(image,image,CV_BGR2BGRA);
                    }
                    //image is now 4 channel, unknown depth but not 8 bit
                    if(_image.depth()>=5){//float
                        image.convertTo(cm,CV_8UC4,255.0);
                    }else if(image.depth()>=2){//0-65535
                        image.convertTo(cm,CV_8UC4,1/256.0);
                    }
                }
                image=cm;
            }
        }
        CV_Assert(image.type()==CV_8UC4);
    }
    //change input image to a texture
    //ArrayTexture tex(image, cvStream);
    simpleTex(image,cvStream);
    cudaTextureObject_t& texObj=*((cudaTextureObject_t*)(char*)_texObj);
//     cudaTextureObject_t texObj=simpleTex(image,cvStream);
//     cudaSafeCall( cudaDeviceSynchronize() );

    //find projection matrix from cost volume to image (3x4)
    Mat viewMatrixImage;
    RTToP(R,T,viewMatrixImage);
    Mat cameraMatrixTex(3,4,CV_64FC1);
    cameraMatrixTex=0.0;
    cameraMatrix.copyTo(cameraMatrixTex(Range(0,3),Range(0,3)));
    cameraMatrixTex(Range(0,2), Range(2,3)) += 0.5;//add 0.5 to x,y out //removing causes crash

    Mat imFromWorld=cameraMatrixTex*viewMatrixImage;//3x4
    Mat imFromCV=imFromWorld*projection.inv();
    imFromCV.colRange(2,3)*=depthStep;
    assert(baseImage.isContinuous());
    assert(lo.isContinuous());
    assert(hi.isContinuous());
    assert(loInd.isContinuous());
    
    //for each slice
    for(int y=0; y<rows; y++){
        //find projection from slice to image (3x3)
        double *p = (double*)imFromCV.data;
        m33 sliceToIm={  p[0], p[2], p[3]+y*p[1],
                         p[4], p[6], p[7]+y*p[5],
                         p[8], p[10], p[11]+y*p[9]};

    }
    double *p = (double*)imFromCV.data;
    m34 persp;
    for(int i=0;i<12;i++) persp.data[i]=p[i];
#define CONST_ARGS rows, cols, layers, rows*cols, \
            hits,  data, (float*) (lo.data), (float*) (hi.data), (float*) (loInd.data),\
            (float3*) (baseImage.data), (float*)baseImage.data, texObj
    float w=count+++initialWeight;//fun parse
    w/=(w+1); 
    assert(localStream);
    globalWeightedBoundsCostCaller(persp,w,CONST_ARGS);
}

CostVolume::~CostVolume(){
    cudaArray*& cuArray=*((cudaArray**)(char*)_cuArray);
    cudaTextureObject_t& texObj=*((cudaTextureObject_t*)(char*)_texObj);
    if (cuArray){
        cudaFreeArray(cuArray);
    }
    if (texObj){
        cudaDestroyTextureObject(texObj);
    }
    
}

