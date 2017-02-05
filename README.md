OpenDTAM
========

An open source implementation of DTAM
On Ubuntu, you need the qtbase5-dev and libopencv-dev packages

## Build Instructions On Ubuntu 12.04 LTS

For building `OpenDTAM`, here is a brief instruction on Ubuntu 12.04 LTS.

### Install dependencies

#### qtbase5-dev

    sudo apt-add-repository ppa:ubuntu-sdk-team/ppa
    sudo apt-get update
    sudo apt-get install qtbase5-dev

#### libopencv-dev

    sudo apt-get install libopencv-dev

#### boost

    sudo apt-get install libboost1.48-all-dev

#### ROS
Add the path where you cloned openDTAM to the ROS_PACKAGE_PATH environment variable. To do this, modify your .bashrc and add at the bottom the following line (replace PATH_TO_PARENT_OF_ORB_SLAM):

export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:PATH_TO_PARENT_OF_ORB_SLAM

### Build OpenDTAM

	**If you use ROS Indigo, remove the depency of opencv2 in the manifest.xml.**

    cd OpenDTAM/Cpp
    mkdir Build
    cmake ..
    make

### Run OpenDTAM

    ./a.out

### Trouble Shooting

Running  "pkg-config --modversion opencv" will tell you what version you have. Hopefully it is 
close to 2.4.9.

You may have problems with the versions of the dependencies, if so you may be able to resolve them by installing the required ones according to the messages output by `cmake`.

The `Trajectory_30_seconds` directory may reside in different path in your system, you can modify them in `testprog.cpp` before running `make`.

Or if none of this works compile OpenCV 2.4.9 from source on your machine.  
Then send me an email with the output of cmake (something like " -- Detected version of GNU GCC: 46 (406) .......")  
and also the output of   
"cmake -L"  
and I can tell you what you need to do.
