#include <opencv2/core/core.hpp>
#include <stdio.h>

#include "convertAhandaPovRayToStandard.h"
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

void App_main(const cv::FileStorage& settings_file)
{
    pthread_setname_np(pthread_self(), "App_main");

    const int   numImg      = 50;
    const int   imagesPerCV = settings_file["imagesPerCostVolume"];
    const int   layers      = settings_file["costVolumeLayers"];
    const float near        = settings_file["nearInverseDistance"];
    const float far         = settings_file["farInverseDistance"];
    
    const float thetaStart  = settings_file["thetaStart"];
    const float thetaMin    = settings_file["thetaMin"];
    const float thetaStep   = settings_file["thetaStep"];
    const float epsilon     = settings_file["epsilon"];
    const float lambda      = settings_file["lambda"];
    
    const int denoiserItrs  = settings_file["denoiserItrs"];
    
    float fx, fy, cx, cy;
    fx = settings_file["Camera.fx"];
    fy = settings_file["Camera.fy"];
    cx = settings_file["Camera.cx"];
    cy = settings_file["Camera.cy"];
    
    // setup camera matrix
    Mat cameraMatrix = (Mat_<double>(3,3) <<  fx,  0.0,  cx,
                                             0.0,   fy,  cy,
                                             0.0,  0.0, 1.0);
    
    char filename[500];
    Mat image, R, T;

    CostVolume costvolume;
    bool is_costvolume_initialized = false;
    
    CudaMem cret(480, 640, CV_32FC1);
    Mat ret = cret.createMatHeader(); // a place to return downloaded images to

    cv::gpu::Stream s;
    int idx = 0;
    for(int i=0; i < numImg; i++)
    {
        sprintf(filename, "../Trajectory_30_seconds/scene_%03d.png", i);
        printf("Opening: %s \n", filename);
        imread(filename, -1).convertTo(image, CV_32FC3, 1.0/65535.0);

        convertAhandaPovRayToStandard("../Trajectory_30_seconds",
                                      i, 
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
        else { //Attach optimizer
            Optimizer optimizer(costvolume, thetaStart, thetaMin, thetaStep, epsilon, lambda);
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
              for(int i = 1; i <= denoiserItrs; i++)
                d=denoiser(a, optimizer.epsilon, optimizer.getTheta());
              doneOptimizing=optimizer.optimizeA(d,a);
              
              d.download(ret);
              pfShow("D function", ret, 0, cv::Vec2d(0, layers));
            } while(!doneOptimizing);

            // cv::waitKey(0);

            optimizer.lambda=0.01f;
            optimizer.optimizeA(d,a);
            optimizer.cvStream.waitForCompletion();

            // reset costvolume
            is_costvolume_initialized = false;
            
            idx += imagesPerCV;
        }
        s.waitForCompletion(); // so we don't lock the whole system up forever
    }
    s.waitForCompletion();
    Stream::Null().waitForCompletion();
}

int main(int argc, char** argv) {
    if(argc < 2) {
      cout << "\nUsage: executable_name path/to/settings_file \n";
      exit(0);
    }
    
    string settings_filename = argv[1];
    cv::FileStorage settings_file(settings_filename, cv::FileStorage::READ);

    initGui();
    App_main(settings_file);
    ImplThread::stopAllThreads();
    return 0;
}
