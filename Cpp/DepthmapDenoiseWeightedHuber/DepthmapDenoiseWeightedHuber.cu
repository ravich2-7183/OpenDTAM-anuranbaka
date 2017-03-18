#include <opencv2/gpu/device/common.hpp> //for cudaSafeCall
#include <opencv2/core/core.hpp> //for CV_Assert
#include "DepthmapDenoiseWeightedHuber.cuh"

namespace cv { namespace gpu { namespace device { namespace dtam_denoise{

static __global__ void computeG(float* g, float* img, int w, int h, float alpha=3.5f, float beta=1.0f)
{
  // thread coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int i  = (y * w + x);

  // gradients gx := $\partial_{x}^{+}img$ computed using forward differences
  float gx = (x==w-1)? 0.0f : img[i+1] - img[i];
  float gy = (y==h-1)? 0.0f : img[i+w] - img[i];

  g[i] = expf(-alpha*powf(sqrtf(gx*gx + gy*gy), beta));
}

// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// Scharr gradient kernel
static __global__ void computeGScharr(float* g, float* img, int w, int h, float alpha=3.5f, float beta=1.0f)
{
    // Calculate texture coordinates
    float x = (float) (blockIdx.x * blockDim.x + threadIdx.x);
    float y = (float) (blockIdx.y * blockDim.y + threadIdx.y);
    
    const int i = (int)(y * w + x);
    
    // Scharr kernels (combines Gaussian smoothing and differentiation)
    /*  kx           ky
       -3 0  3       -3 -10 -3
      -10 0 10        0   0  0
       -3 0  3        3  10  3
    */

    // Out of border references are clamped to [0, N-1]
    float gx, gy;
    gx = -3.0f * tex2D(texRef, x-1, y-1) +
          3.0f * tex2D(texRef, x+1, y-1) +
         10.0f * tex2D(texRef, x-1, y  ) +
        -10.0f * tex2D(texRef, x+1, y  ) +
         -3.0f * tex2D(texRef, x-1, y+1) +
          3.0f * tex2D(texRef, x+1, y+1) ;

    gy = -3.0f * tex2D(texRef, x-1, y-1) +
         -3.0f * tex2D(texRef, x+1, y-1) +
        -10.0f * tex2D(texRef, x  , y-1) +
         10.0f * tex2D(texRef, x  , y+1) +
          3.0f * tex2D(texRef, x-1, y+1) +
          3.0f * tex2D(texRef, x+1, y+1) ;
    
    g[i] = expf(-alpha*powf(sqrtf(gx*gx + gy*gy), beta));
}

void computeGCaller(float* img, float* g,
                    int width, int height, int pitch,
                    float alpha, float beta, bool useScharr)
{
  // TODO set dimBlock based on warp size
  dim3 dimBlock(16, 16);
  dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
               (height + dimBlock.y - 1) / dimBlock.y);

  if(useScharr) {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    // Set texture reference parameters
    texRef.normalized     = false;
    texRef.addressMode[0] = cudaAddressModeClamp; // out of border references return first or last element, 
    texRef.addressMode[1] = cudaAddressModeClamp; // this is good enough for Sobel/Scharr filter
    texRef.filterMode     = cudaFilterModeLinear;

    // Bind the array to the texture reference
    size_t offset;
    cudaBindTexture2D(&offset, texRef, img, channelDesc, width, height, pitch);
 
    // Invoke kernel
    computeGScharr<<<dimGrid, dimBlock>>>(g, img, width, height, alpha, beta);
    cudaDeviceSynchronize();
    cudaUnbindTexture(texRef);
    cudaSafeCall( cudaGetLastError() );
  }
  else {
    computeG<<<dimGrid, dimBlock>>>(g, img, width, height, alpha, beta);
    cudaDeviceSynchronize();
    cudaSafeCall( cudaGetLastError() );
  }
}

static __global__ void update_q(float *g, float *a,  // const input
                                float *q, float *d,  // input  q, d
                                int w, int h, // dimensions: width, height
                                float sigma_q, float sigma_d, float epsilon, float theta // parameters
                                )
{
  // thread coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int i  = (y * w + x);
  const int wh = (w*h);

  // gradients dd_x := $\partial_{x}^{+}d$ computed using forward differences
  float dd_x = (x==w-1)? 0.0f : d[i+1] - d[i];
  float dd_y = (y==h-1)? 0.0f : d[i+w] - d[i];

  float qx = (q[i]    + sigma_q*g[i]*dd_x) / (1.0f + sigma_q*epsilon);
  float qy = (q[i+wh] + sigma_q*g[i]*dd_y) / (1.0f + sigma_q*epsilon);

  // reproject q **element-wise**
  // if the whole vector q had to be reprojected, a tree-reduction sum would have been required
  float maxq = fmaxf(1.0f, sqrtf(qx*qx + qy*qy));
  q[i]    = qx / maxq;
  q[i+wh] = qy / maxq;
}

static __global__ void update_d(float *g, float *a,  // const input
                                float *q, float *d,  // input  q, d
                                int w, int h, // dimensions: width, height
                                float sigma_q, float sigma_d, float epsilon, float theta // parameters
                                )
{
  // thread coordinates
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  const int i  = (y * w + x);
  const int wh = (w*h);

  // div_q computed using backward differences
  float dqx_x = (x==0)? q[i]    - q[i+1]    : q[i]    - q[i-1];
  float dqy_y = (y==0)? q[i+wh] - q[i+wh+w] : q[i+wh] - q[i+wh-w];
  float div_q = dqx_x + dqy_y;

  d[i]  = (d[i] + sigma_d*(g[i]*div_q + a[i]/theta)) / (1.0f + sigma_d/theta);
}

void update_q_dCaller(float *g, float *a,  // const input
                      float *q,  float *d,  // input q, d
                      int width, int height, // dimensions
                      float sigma_q, float sigma_d, float epsilon, float theta // parameters
                      )
{
  dim3 dimBlock(16, 16);
  dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
               (height + dimBlock.y - 1) / dimBlock.y);

  update_q<<<dimGrid, dimBlock>>>(g, a,  // const input
                                  q, d,  // input  q, d
                                  width, height, // dimensions: width, height
                                  sigma_q, sigma_d, epsilon, theta // parameters
                                  );
  cudaDeviceSynchronize();
  cudaSafeCall( cudaGetLastError() );

  update_d<<<dimGrid, dimBlock>>>(g, a,  // const input
                                  q, d,  // input  q, d
                                  width, height, // dimensions: width, height
                                  sigma_q, sigma_d, epsilon, theta // parameters
                                  );
  cudaDeviceSynchronize();
  cudaSafeCall( cudaGetLastError() );
}

}}}}
