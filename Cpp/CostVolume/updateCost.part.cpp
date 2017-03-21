//#ifdef COST_CPP_SUBPARTS
#include "updateCost.part.hpp"
#include "CostVolume/utils/reproject.hpp"

//debugs
#include <iostream>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include "tictoc.h"
#include "graphics.hpp"

using namespace std;
static inline float fastabs(const float& foo){
    return fabs(foo);
}

void Cost::updateCostL1(const cv::Mat& image, const cv::Matx44d& currentCameraPose)
{
    imageNum++;
    cv::Mat newLo(rows,cols,CV_32FC1,1000.0);
    newLo=1000.0;
    
    cv::Mat newHi(rows,cols,CV_32FC1,0);
    for(int n=0; n < depth.size(); ++n){
       // tic();
        cv::Mat_<cv::Vec3f> plane;
        cv::Mat_<uchar> mask;
        reproject(cv::Mat_<cv::Vec3f>(image), cameraMatrix, pose, currentCameraPose, depth[n], plane, mask);//could be as fast as .00614 using resize instead. Currently runs at .0156s, or about twice as long
        size_t end=image.rows*image.cols*image.channels();
        size_t lstep=layers;
        #ifdef DTAM_COST_DEBUG
        float* pdata;
#else
        const float* pdata;
#endif
        pdata=(float*)(plane.data);
        const float* idata=(float*)(baseImage.data);
        float* cdata=data+n;
        float* hdata=hit+n;
        float* xdata=(float*)(newHi.data);
        char*  mdata=(char*)(mask.data);
        
        for (size_t i=0, moff=0,coff=0,p=0;  i<end; p++, moff+=3, i+=3, coff+=lstep){//.0055 - .0060 s 
            //std::cout<<mdata[moff]<<std::endl;
            if(mdata[moff]){
                float v1=fastabs(pdata[i]-idata[i]);
                float v2=fastabs(pdata[i+1]-idata[i+1]);
                float v3=fastabs(pdata[i+2]-idata[i+2]);
                float h=hdata[coff]+1;
                float ns=cdata[coff]*(1-1/h)+(v1+v2+v3)/h;
                
                hdata[coff]=h;
                cdata[coff]=ns;
                
               // std::cout<<ns<<std::endl;
            }
#ifdef DTAM_COST_DEBUG
            {//debug see the cost
                pdata[i]=cdata[coff];
                pdata[i+1]=cdata[coff];
                pdata[i+2]=cdata[coff];
            }
#endif
        }
#ifdef DTAM_COST_DEBUG
        {  //debug
           pfShow( "Cost Volume Slice", plane,0,cv::Vec2d(0,.5));
        }
#endif
    }
}

void Cost::updateCostL2(const cv::Mat& image,
                        const cv::Matx44d& currentCameraPose)
{
    if (image.type()==CV_32FC3){
        cv::Mat newLo(rows,cols,CV_32FC1,1000.0);
        newLo=1000.0;
        cv::Mat lInd(rows,cols,CV_32SC1);
        lInd=255;
        
        cv::Mat newHi(rows,cols,CV_32FC1,0);
        for(int n=0; n < depth.size(); ++n){
            
            cv::Mat_<cv::Vec3f> _img(image);
            cv::Mat_<cv::Vec3f> plane;
            cv::Mat_<uchar> mask;
            reproject(_img, cameraMatrix, pose, currentCameraPose, depth[n], plane, mask);
            size_t end=image.rows*image.cols*image.channels();
            size_t lstep=layers;
            float* pdata=(float*)(plane.data);
            float* idata=(float*)(baseImage.data);
            float* cdata=data+n;
            float* hdata=hit+n;
            float* ldata=(float*)(newLo.data);
            
            float* xdata=(float*)(newHi.data);
            char*  mdata=(char*)(mask.data);
            
            //size_t moff=0;
            for (size_t i=0, moff=0,coff=0,p=0;  i<end; p++, moff+=3, i+=3, coff+=lstep){
                if (n==0){
                    ldata[p]=255.0;
                }
                //std::cout<<mdata[moff]<<std::endl;
                if(mdata[moff]){
                    float v1=pdata[i]-idata[i];
                    float v2=pdata[i+1]-idata[i+1];
                    float v3=pdata[i+2]-idata[i+2];
                    float ns=cdata[coff]+v1*v1+v2*v2+v3*v3;
                    cdata[coff]=ns;
                    
                    if(ldata[p]>(ns/hdata[coff])){
                        ldata[p]=ns/hdata[coff];
                        
                        ((int*)(lInd.data))[p]=n;
                    }
                    
                    hdata[coff]+=1.0;
                }
            }
        }
    }
    else if (image.type()==CV_32FC1){
        for(int n=0; n < depth.size(); ++n){
            cv::Mat_<float> _img(image);
            cv::Mat_<float> plane;
            cv::Mat_<uchar> mask;
            reproject(_img, cameraMatrix, pose, currentCameraPose, depth[n], plane, mask);
            size_t end=image.rows*image.cols;
            size_t lstep=layers;
            float* pdata=(float*)(plane.data);
            float* idata=(float*)(baseImage.data);
            float* cdata=((float*)data)+n;
            float* hdata=((float*)hit)+n;
            char*  mdata=(char*)(mask.data);
            
            //size_t moff=0;
            for (size_t i=0, moff=0,coff=0;  i<end;  moff++, i++, coff+=lstep){
                //std::cout<<mdata[moff]<<std::endl;
                if(mdata[moff]){
                    float v1=pdata[i]-idata[i];
                    cdata[coff]+=v1*v1;
                    hdata[coff]++;
                }
            }   
        }
    }
    else{
        std::cout<<"Error, Unsupported Type!"<<std::endl;
        assert(false);
    }
}

void Cost::updateCostL1(const cv::Mat& image, const cv::Mat& R, const cv::Mat& Tr)
{
    updateCostL1(image,convertPose(R,Tr));
}

void Cost::updateCostL2(const cv::Mat& image, const cv::Mat& R, const cv::Mat& Tr)
{
    updateCostL2(image,convertPose(R,Tr));
}
//#endif
