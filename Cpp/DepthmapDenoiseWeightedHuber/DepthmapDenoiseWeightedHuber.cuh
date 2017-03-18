
#ifndef COSTVOLUME_CUH
#define COSTVOLUME_CUH
#include <opencv2/gpu/device/common.hpp>//for cudaStream_t
namespace cv { namespace gpu { namespace device { namespace dtam_denoise{

void computeGCaller(float* img, float* g,
                    int width, int height, int pitch,
                    float alpha=3.5f, float beta=1.0f, bool useScharr=false);

void update_q_dCaller(float *g, float *a,  // const input
                      float *q, float *d,  // input q, d
                      int width, int height, // dimensions
                      float sigma_q, float sigma_d, float epsilon, float theta // parameters
                      );



}}}}
#endif
