#ifndef CONVERTAHANDAPOVRAYTOSTANDARD_H_INCLUDED
#define CONVERTAHANDAPOVRAYTOSTANDARD_H_INCLUDED
#include <opencv2/opencv.hpp>
void convertAhandaPovRayToStandard(const char * filepath,
                                   int imageNumber,
                                   cv::Mat& R,
                                   cv::Mat& T);
#endif // CONVERTAHANDAPOVRAYTOSTANDARD_H_INCLUDED
