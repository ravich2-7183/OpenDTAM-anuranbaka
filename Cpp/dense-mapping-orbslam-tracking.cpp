#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <tf/transform_datatypes.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/image_encodings.h>
#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>

#include "CostVolume/utils/reproject.hpp"
#include "CostVolume/utils/reprojectCloud.hpp"
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

class DenseMapper
{
  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  tf::TransformListener tf_listener_;
  tf::StampedTransform transform_;

  const int   images_per_cost_volume_;
  const int   layers_;
  const float near_;
  const float far_;

  CostVolume costvolume_;
  bool is_costvolume_initialized_;
  Stream gpu_stream_;

  Mat image_;
  const int rows_, cols_;
  double fps_;
  Mat camera_matrix_;
  cv_bridge::CvImagePtr input_bridge_;

  vector<Mat> images_, Rs_cv_, Ts_cv_;

  //a place to return downloaded images to
  Mat ret_;
  CudaMem cret_;
  
  // TODO: remove this after comparing tf from ahanda and orb_slam
  ofstream tf_out_;
  int tf_idx_;

public:
  DenseMapper(int rows, int cols, cv::FileStorage& settings_file)
    : nh_(),
      rows_(rows),
      cols_(cols),
      cret_(rows, cols, CV_32FC1),
      images_per_cost_volume_(5), 
      layers_(32),
      near_(0.010f),
      far_(0.0f),
      is_costvolume_initialized_(false),
      tf_out_("../orb_slam_tf.txt", ofstream::out),
      tf_idx_(0)
  {
    ret_=cret_.createMatHeader();

    // setup camera matrix
    double fx = settings_file["Camera.fx"];
    double fy = settings_file["Camera.fy"];
    double cx = settings_file["Camera.cx"];
    double cy = settings_file["Camera.cy"];

    fps_ = settings_file["Camera.fps"];

    camera_matrix_ = (Mat_<double>(3,3) << fx , 0.0, cx,
                                           0.0, fy , cy,
                                           0.0, 0.0, 1.0);

    // setup display windows
    ret_ = Mat::zeros(rows, cols, CV_32FC1);
    pfShow("A function",       ret_, 0, cv::Vec2d(0, layers_));
    pfShow("D function",       ret_, 0, cv::Vec2d(0, layers_));
    pfShow("A function loose", ret_, 0, cv::Vec2d(0, layers_));
    pfShow("Predicted Image",  ret_, 0, cv::Vec2d(0,1));
    pfShow("Actual Image",     ret_);
  }
    
  void Run()
  {
    sub_ = nh_.subscribe("/camera/image_raw", 1, &DenseMapper::imageCb, this);
    ros::spin();
  }
      
  void imageCb(const sensor_msgs::ImageConstPtr& image_msg)
  {
    try {
      input_bridge_ = cv_bridge::toCvCopy(image_msg); // TODO: use cv_bridge::toCvShare instead?
      Mat im_temp1 = input_bridge_->image; // TODO: is this copy required?
      Mat im_temp2;
      im_temp1.convertTo(im_temp2, CV_32FC3, 1.0/65535.0);
      if(im_temp2.cols != cols_ || im_temp2.rows != rows_)
          cv::resize(im_temp2, image_, Size(cols_, rows_));
      else
        image_ = im_temp2;
      // TODO: comment or remove this line
      ROS_INFO("image_ size = %d x %d \n", image_.cols, image_.rows);
      images_.push_back(image_.clone());
    }
    catch (cv_bridge::Exception& ex) {
      ROS_ERROR("[DenseMapper] Failed to convert image: \n%s", ex.what());
      return;
    }

    try {
      ros::Time acquisition_time = image_msg->header.stamp;
      ros::Duration timeout(1.0 / fps_);
      tf_listener_.waitForTransform("/ORB_SLAM/World", "/ORB_SLAM/Camera", 
                                    acquisition_time, timeout);
      tf_listener_.lookupTransform("/ORB_SLAM/World", "/ORB_SLAM/Camera", 
                                   acquisition_time, transform_);
    }
    catch (tf::TransformException& ex) {
      ROS_WARN("[DenseMapper] TF exception: \n%s", ex.what());
      return;
    }

    tf::Matrix3x3 R = transform_.getBasis();
    tf::Vector3   T = transform_.getOrigin();

    cv::Mat Rcv = (Mat_<double>(3,3) << R[0].x(), R[0].y(), R[0].z(), 
                                        R[1].x(), R[1].y(), R[1].z(), 
                                        R[2].x(), R[2].y(), R[2].z());

    
    cv::Mat Tcv = (Mat_<double>(3,1) << T.x(), T.y(), T.z());

    tf_out_ << "Rcv[" << tf_idx_ << "] = " << endl << Rcv << endl;
    tf_out_ << "Tcv[" << tf_idx_ << "] = " << endl << Tcv << endl;
    tf_idx_++;

    // cout << "Rcv = "<< endl << " "  << Rcv << endl;
    // cout << "Tcv = "<< endl << " "  << Tcv << endl;
    
    Rs_cv_.push_back(Rcv.clone());
    Ts_cv_.push_back(Tcv.clone());
    
    if(!is_costvolume_initialized_) {
      CostVolume costvolume(image_, (FrameID)0, layers_,
                            near_, far_,
                            Rcv, Tcv, camera_matrix_);
      costvolume_ = costvolume;
      is_costvolume_initialized_ = true;
    }
      
    if(costvolume_.count < images_per_cost_volume_){
      // update costvolume_ & increment costvolume_.count
      costvolume_.updateCost(image_, Rcv, Tcv); 
    }
    else{
      // Attach optimizer & estimate depth map
      Optimizer optimizer(costvolume_);
      optimizer.initOptimization();
      gpu_stream_=optimizer.cvStream;
        
      Ptr<DepthmapDenoiseWeightedHuber> dp = createDepthmapDenoiseWeightedHuber(costvolume_.baseImageGray,
                                                                                costvolume_.cvStream);
      DepthmapDenoiseWeightedHuber& denoiser=*dp;
      denoiser.cacheGValues();
        
      GpuMat a(costvolume_.loInd.size(), costvolume_.loInd.type());
      costvolume_.cvStream.enqueueCopy(costvolume_.loInd, a);
      GpuMat d;
        
      bool doneOptimizing; int Acount=0; int QDcount=0;
      do{
        for(int i = 0; i < 10; i++) {
          d=denoiser(a, optimizer.epsilon, optimizer.getTheta());
          QDcount++;
        }
        d.download(ret_);
        pfShow("D function", ret_, 0, cv::Vec2d(0, layers_));
          
        doneOptimizing=optimizer.optimizeA(d,a);
        Acount++;
        a.download(ret_);
        pfShow("A function", ret_, 0, cv::Vec2d(0, layers_));
      }while(!doneOptimizing);
        
      optimizer.lambda=0.01f;
      optimizer.optimizeA(d,a);
      optimizer.cvStream.waitForCompletion();
      a.download(ret_);
      pfShow("A function loose", ret_, 0, cv::Vec2d(0, layers_));
        
      // TODO: use ros mechanisms to reproject the cloud 
      for(int i=0; i < images_per_cost_volume_; i++){
        reprojectCloud(images_[i], images_[0],
                       optimizer.depthMap(),
                       RTToP(Rs_cv_[0], Ts_cv_[0]),
                       RTToP(Rs_cv_[i], Ts_cv_[i]),
                       camera_matrix_);
      }
        
      // reset costvolume_
      is_costvolume_initialized_ = false;
      images_.clear();
      Rs_cv_.clear();
      Ts_cv_.clear();
    }
    gpu_stream_.waitForCompletion(); // so we don't lock the whole system up forever
    Stream::Null().waitForCompletion();
  } // DenseMapper::imageCb close
}; // class DenseMapper

void myExit(){
  ImplThread::stopAllThreads();
}

int main( int argc, char** argv ) {
  if(argc < 2) {
    cout << "\nUsage: executable_name path/to/settings_file \n";
    exit(0);
  }

  string settings_filename = argv[1];
  cv::FileStorage settings_file(settings_filename, cv::FileStorage::READ);

  int rows = settings_file["Camera.rows"];
  int cols = settings_file["Camera.cols"];
  rows = (rows%32 != 0)? 32*(rows/32) : rows ; // int rounding to meet OpenDTAM requirements
  cols = (cols%32 != 0)? 32*(cols/32) : cols ; // int rounding to meet OpenDTAM requirements
  cout << "\n DenseMapper: rows = " << rows << ", cols = " << cols << "\n";

  ros::init(argc, argv, "dense_mapper");
  ros::start();
    
  initGui();

  DenseMapper dense_mapper(rows, cols, settings_file);
  boost::thread dense_mapper_thread(&DenseMapper::Run, &dense_mapper);
  dense_mapper_thread.join();

  myExit();
  ros::shutdown();
  return 0;
}
