#include <opencv2/core/core.hpp>
#include <stdio.h>

#include "convertAhandaPovRayToStandard.h"
#include "CostVolume/Cost.h"
#include "CostVolume/CostVolume.hpp"
#include "Optimizer/Optimizer.hpp"
#include "DepthmapDenoiseWeightedHuber/DepthmapDenoiseWeightedHuber.hpp"
#include "graphics.hpp"
#include "set_affinity.h"
#include "utils/utils.hpp"
#include "tictoc.h"

//A test program to make the mapper run
using namespace cv;
using namespace cv::gpu;
using namespace std;

void App_main()
{
    pthread_setname_np(pthread_self(), "App_main");

    const int    numImg              = 200;
    const int    imagesPerCV         = 5;
    const int    layers              = 32;
    const float  near                = 0.010f;
    const float  far                 = 0.0f;

    char filename[500];
    Mat cameraMatrix;
    Mat image, R, T;

    CostVolume costvolume;
    bool is_costvolume_initialized = false;
    
    CudaMem cret(480, 640, CV_32FC1);
    Mat ret = cret.createMatHeader(); // a place to return downloaded images to

    cv::gpu::Stream s;
    int idx = 0;
    for(int i=0; i < numImg; i++){
        sprintf(filename, "../Trajectory_30_seconds/scene_%03d.png", i);
        printf("Opening: %s \n", filename);
        imread(filename, -1).convertTo(image, CV_32FC3, 1.0/65535.0);

        convertAhandaPovRayToStandard("../Trajectory_30_seconds",
                                      i, cameraMatrix,
                                      R, T);

        if(!is_costvolume_initialized) {
          CostVolume costvolume_temp(image, (FrameID)0, layers,
                                     near, far,
                                     R, T, cameraMatrix);
          costvolume = costvolume_temp;
          is_costvolume_initialized = true;
        }

        if(costvolume.count < imagesPerCV){
            costvolume.updateCost(image, R, T); // increments costvolume.count
        }
        else{ //Attach optimizer
            Optimizer optimizer(costvolume);
            optimizer.initOptimization();
            s=optimizer.cvStream;
            
            Ptr<DepthmapDenoiseWeightedHuber> dp = createDepthmapDenoiseWeightedHuber(costvolume.baseImageGray,
                                                                                      costvolume.cvStream);
            DepthmapDenoiseWeightedHuber& denoiser=*dp;
            denoiser.cacheGValues();

            GpuMat a(costvolume.loInd.size(), costvolume.loInd.type());
            costvolume.cvStream.enqueueCopy(costvolume.loInd, a);
            GpuMat d;

            bool doneOptimizing;
            do {
              for(int i = 0; i < 10; i++) {
                d=denoiser(a, optimizer.epsilon, optimizer.getTheta()); }
              d.download(ret);
              pfShow("D function", ret, 0, cv::Vec2d(0, layers));
              
              doneOptimizing=optimizer.optimizeA(d,a);
              a.download(ret);
              pfShow("A function", ret, 0, cv::Vec2d(0, layers));
            }while(!doneOptimizing);

            optimizer.lambda=0.01f;
            optimizer.optimizeA(d,a);
            optimizer.cvStream.waitForCompletion();

            // reset costvolume
            idx += imagesPerCV;
            is_costvolume_initialized = false;
        }
        s.waitForCompletion(); // so we don't lock the whole system up forever
    }
    s.waitForCompletion();
    Stream::Null().waitForCompletion();
}

int main( int argc, char** argv ){
    initGui();
    App_main();
    ImplThread::stopAllThreads();
    return 0;
}
