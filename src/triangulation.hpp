#ifndef TRIANGULATION_HPP
#define TRIANGULATION_HPP

#include <cstdio>
#include <opencv2/opencv.hpp>

// triangulation (Proj -> Cam = 0) , (Cam -> Proj = 1)
enum TriangulationMode { TR_PROJ = 0, TR_CAM = 1 };

// strings pour les noms de fichier
enum TriangulationFiles { IDX_TR_MASK = 0, IDX_TR_DATA = 1, IDX_TR_PARC = 2, IDX_TR_PARP = 3 };

class Triangulation {

    // projector
    cv::Mat internes_proj;
    cv::Mat rotation_proj;
    cv::Mat translation_proj;
    cv::Mat distCoeffs_proj;
    cv::Mat poseMatrix_proj;

    // camera
    cv::Mat internes_cam;
    cv::Mat rotation_cam;
    cv::Mat translation_cam;
    cv::Mat distCoeffs_cam;
    cv::Mat poseMatrix_cam;

    // filenames
    const char *fn_tr_mask;
    const char *fn_tr_data;
    const char *fn_tr_parc;
    const char *fn_tr_parp;

  public:
    Triangulation();
    ~Triangulation();

    void triangulate(const cv::Mat &lutCam, const cv::Mat &lutProj);

    void setPathT(int idx, const std::string &path, const char *filename);

  private:
    int initMat(const std::string &file, cv::Mat &internes, cv::Mat &rotation, cv::Mat &translation,
                cv::Mat &distCoeffs);

    int matrixCorr(cv::Mat &pointsLut, cv::Mat &pointsCorr, const cv::Mat &lutSrc,
                   const cv::Mat &lutDst);

    int saveMat(const cv::Mat &point4D);

    int lut2corr(const cv::Mat &lutSrc, const cv::Mat &internesSrc, const cv::Mat &distCoeffsSrc,
                 cv::Mat &pointsUndSrc, const cv::Mat &lutDst, const cv::Mat &internesDst,
                 const cv::Mat &distCoeffsDst, cv::Mat &pointsUndDst);

    void composePoseMatrix(cv::Mat &poseMatrix, const cv::Mat &rotation,
                           const cv::Mat &translation);

    cv::Mat undistortMatrix(const cv::Mat &pointsInput, const cv::Mat &internes,
                            const cv::Mat &distCoeffs);
};

#endif // TRIANGULATION_HPP
