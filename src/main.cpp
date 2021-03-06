﻿//#include <QCoreApplication>
//#include <sys/time.h>
//#include <sys/stat.h>
//#include <unistd.h>

#include <opencv2/opencv.hpp>

#include <chrono>
#include <thread>

#include "util.hpp"
#include "leopard.hpp"
#include "paths.hpp"
#include "triangulation.hpp"

using namespace cv;
using namespace std;
using namespace util;

void testLeopardSeb(const std::vector<cv::Mat>& imagesCam, const std::vector<cv::Mat>& imagesProj) {
    printf("----- test leopard seb -----\n");

    printf("sizeof char %d\n", (int)sizeof(char));
    printf("sizeof short %d\n", (int)sizeof(short));
    printf("sizeof int %d\n", (int)sizeof(int));
    printf("sizeof long %d\n", (int)sizeof(long));
    printf("sizeof long long %d\n", (int)sizeof(long long));

	string pathscan = "";
	int nb = imagesCam.size();

    // setup les output
	Leopard leo;
    leo.setPathL(IDX_SCAN_MASKC, pathscan, "maskcam.png");
    leo.setPathL(IDX_SCAN_MEANC, pathscan, "meancam.png");
    leo.setPathL(IDX_SCAN_MASKP, pathscan, "maskproj.png");
    leo.setPathL(IDX_SCAN_MEANP, pathscan, "meanproj.png");

    leo.computeMask(1, imagesCam, nb, 1.45, 5.0, 1, -1, -1, -1,-1); 
    leo.computeCodes(1, LEOPARD_SIMPLE, imagesCam);

    leo.computeMask(0, imagesProj, nb, 1.45, 5.0, 1, -1, -1, -1,-1);
    leo.computeCodes(0, LEOPARD_SIMPLE, imagesProj);

    // quelques stats
    // L->statsCodes(1);
    // L->statsCodes(0);

    leo.prepareMatch();
    // L->forceBrute();

    for(int i = 0; i < 10; i++) {
        printf("--- %d ---\n", i);
        leo.doLsh(0, 0);
        // L->doHeuristique();
    }

    cv::Mat mixCam, lutCam;
    cv::Mat mixProj, lutProj;

    std::tie(lutCam, mixCam)   = leo.makeLUT(1);
    std::tie(lutProj, mixProj) = leo.makeLUT(0);

    imwrite("lutcam.png", lutCam);
    imwrite("lutproj.png", lutProj);

    printf("----- done -----\n");
}


enum CostDirection {
	Forward = 1,
	BackWard = -1
};

double computeCost(Leopard& leo, bool forward, const std::vector<cv::Mat>& imagesCamDecal, const std::vector<cv::Mat>& imagesProj) {
	int nb = imagesProj.size();
	std::vector<cv::Mat> imagesProjMix(nb);

	int start = 0;
	int step = 1;
	int bound = nb - 1;

	if (!forward) {
		printf("match avec la suivante! \n");
		start = nb - 1;
		step = -1;
		bound = 0;
	}

	leo.prepareMatch();
	for (int i = start; i < bound; i += step) {
		imagesProjMix[i] = imagesProj[i] * 0.5 + imagesProj[i + step] * 0.5;
	}
	imagesProjMix[bound] = imagesProj[bound];

	// QUAD
	leo.computeCodes(1, LEOPARD_QUADRATIC, imagesCamDecal);
	leo.computeCodes(0, LEOPARD_QUADRATIC, imagesProjMix);

	for (int j = 0; j < 10; j++)
		leo.doLsh(0, 0);

	return leo.sumCost();
}

void testLeopardChaima(const std::vector<Mat> &imagesCam, const std::vector<Mat> &imagesProj, const string &namelutC,
                       const string &namelutP, const string &namemixC, const string &namemixP,
                       Mat &lutCam, Mat &lutProj, int sp) {

    printf("----- test leopard chaima -----\n");

    printf("sizeof char %d\n", (int)sizeof(char));
    printf("sizeof short %d\n", (int)sizeof(short));
    printf("sizeof int %d\n", (int)sizeof(int));
    printf("sizeof long %d\n", (int)sizeof(long));
    printf("sizeof long long %d\n", (int)sizeof(long long));

	int nb = imagesCam.size();
    Chronometer chrono;
    Leopard leo;

    // setup les filename
    leo.setPathL(IDX_SCAN_MASKC, path, FN_SCAN_MASKC);
    leo.setPathL(IDX_SCAN_MEANC, path, FN_SCAN_MEANC);
    leo.setPathL(IDX_SCAN_MASKP, path, FN_SCAN_MASKP);
    leo.setPathL(IDX_SCAN_MEANP, path, FN_SCAN_MEANP);

    leo.computeMask(1, imagesCam, nb, 0.65, 5.0, 1, -1, -1, -1, -1); // 815,815+20,815,815+20
    leo.computeCodes(1, LEOPARD_SIMPLE, imagesCam);

    leo.computeMask(0, imagesProj, nb, 1.45, 5.0, 1, -1, -1, -1, -1);
    leo.computeCodes(0, LEOPARD_SIMPLE, imagesProj);

    // Cherche la premiere image de la séquence camera
    leo.prepareMatch();
    int posR = leo.doShiftCodes();

    std::vector<Mat> imagesCamDecal(nb);
    for(int i = 0; i < nb; i++)
        imagesCamDecal[i] = imagesCam[(i + posR) % nb];

    // Cherche le mix des images du projecteur

	int sumCostS = computeCost(leo, true, imagesCamDecal, imagesProj);
	int sumCostP = computeCost(leo, false, imagesCamDecal, imagesProj);



	/*
	std::vector<Mat> imagesProjMix(nb);
    // Match avec l'image suivante
    leo.prepareMatch();
    for(int i = 0; i < nb - 1; i++) {
        imagesProjMix[i] = imagesProj[i] * 0.5 + imagesProj[i + 1] * 0.5;
    }
    imagesProjMix[nb - 1] = imagesProj[nb - 1];

    // QUAD
    leo.computeCodes(1, LEOPARD_QUADRATIC, imagesCamDecal);
    leo.computeCodes(0, LEOPARD_QUADRATIC, imagesProjMix);

    for(int j = 0; j < 10; j++)
        leo.doLsh(0, 0);

    sumCostS = leo.sumCost();

    // Match avec la précédente
    leo.prepareMatch();
    for(int i = nb - 1; i > 0; i--)
        imagesProjMix[i] = imagesProj[i] * 0.5 + imagesProj[i - 1] * 0.5;
    imagesProjMix[0] = imagesProj[0];

    // QUAD
    leo.computeCodes(1, LEOPARD_QUADRATIC, imagesCamDecal);
    leo.computeCodes(0, LEOPARD_QUADRATIC, imagesProjMix);

    for(int j = 0; j < 10; j++)
        leo.doLsh(0, 0);

    sumCostP = leo.sumCost();*/

    printf("\n sumSuivante = %d, sumPrécédente = %d \n", sumCostS, sumCostP);

    double timeSM = chrono.time();

    // Choix du mix
    leo.prepareMatch();

	int start = 0;
	int step = 1;
	int bound = nb - 1;

	if (sumCostS < sumCostP) {
		printf("match avec la suivante! \n");
		start = 0;
		step = 1;
		bound = nb - 1;
	}
	else {
		printf("match avec la précédente! \n");
		start = nb - 1;
		step = -1;
		bound = 0;
	}

	std::vector<Mat> imagesProjMix(nb);
    for(double fct = 0; fct <= 1; fct += 0.1) {
        printf("---------------------- facteur = %.2f \n\n",
                fct);

        for(int i = start; i < bound; i += step)
            imagesProjMix[i] = imagesProj[i] * (1 - fct) + imagesProj[i + step] * fct;

        imagesProjMix[bound] = imagesProj[bound];

        // QUAD
        leo.computeCodes(1, LEOPARD_QUADRATIC, imagesCamDecal);
        leo.computeCodes(0, LEOPARD_QUADRATIC, imagesProjMix);

        // TEST: pas de cumul
        // L->prepareMatch();
        for(int j = 0; j < 20; j++)
            leo.doLsh(sp, int(fct * 255));

        // L->forceBrute(sp,(int) (fct*255));
    }

    chrono.start();
    // L->forceBrute();

    cv::Mat mixCam;
    cv::Mat mixProj;

    std::tie(lutCam, mixCam)   = leo.makeLUT(1);
    std::tie(lutProj, mixProj) = leo.makeLUT(0);

    imwrite(namelutC, lutCam);
    imwrite(namelutP, lutProj);
    imwrite(namemixC, mixCam);
    imwrite(namemixP, mixProj);

    double timeE = chrono.time();

    printf("\n Time Scan = %f \n", timeE);
    printf("\n Time match mix = %f \n", timeSM);
    printf("----- leopard done -----\n");
}

// Capture projected image onto the object using the camera
// Save the images in
// std::vector<Mat>
std::tuple<std::vector<Mat>, bool> capture(int image_number, const std::string &name_format,
                                           int img_width = 1920, int img_height = 1080,
                                           int fps = 30, int sleep_time = 1000) {
    printf("----- Capture -----\n");
    std::vector<Mat> img(image_number);

    VideoCapture cap(1);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, img_width);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, img_height);
    cap.set(CV_CAP_PROP_FPS, fps);

    // Discard first 30 images
    for(int i = 0; i < 30; i++)
        cap >> img[0]; // Démarrer la caméra

    if(!cap.isOpened()) {
        cout << "Camera error" << endl;
        return std::make_tuple(std::vector<Mat>(), false);
    } else {
        cout << "Camera ready" << endl;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));

    Chronometer chrono;
    for(int i = 0; i < image_number; i++) {
        cap >> img[i];
        // resize(img[i], resized[i], Size(640,480)); //(683,384)
    }
    double timeE = chrono.time();

    double xs = cap.get(CV_CAP_PROP_FRAME_WIDTH);
    double ys = cap.get(CV_CAP_PROP_FRAME_HEIGHT);

    cout << "x = " << xs << "   y = " << ys << endl;
    cout << "Time : " << timeE / image_number << endl;
    cout << "Time (" << image_number << " images): " << timeE << endl;
    waitKey(0);

    namedWindow("Display Image", 1);

    for(int i = 0; i < image_number; i++) {
        imwrite(format(name_format.c_str(), i), img[i]);
        imshow("Display Image", img[i]);
        waitKey(30);
    }

    printf("----- Capture done -----\n");
    printf("\n\n");
    return std::make_tuple(img, true);
}

int main(int argc, char *argv[]) {

    const int nbImages = 300;
    std::vector<Mat> img(nbImages);

    // Créer des directory pour stocker les images
#ifdef __linux__
    string dir = "mkdir -p" + path + "Output/scan/lut " + path + "Output/scan/mask " + path +
                 "Output/triangulation";

    system(dir.c_str()); //*/
#endif

    int doCapture   = 0;
    int doScan      = 0;
    int doTriangule = 0;
    int doSp        = 0;
    int elasmi      = 0;
    int roys        = 0;

    // options
    for(int i = 1; i < argc; i++) {
        if(strcmp("-h", argv[i]) == 0) {
            printf("Usage: %s -h\n", argv[0]);
            exit(0);
        } else if(strcmp("-capture", argv[i]) == 0) {
            doCapture = 1;
            continue;
        } else if(strcmp("-scan", argv[i]) == 0) {
            doScan = 1;
            continue;
        } else if(strcmp("-triangule", argv[i]) == 0) {
            doTriangule = 1;
            continue;
        } else if(strcmp("-sp", argv[i]) == 0) {
            doSp = 1;
            continue;
        } else if(strcmp("-roys", argv[i]) == 0) {
            roys = 1;
        } else if(strcmp("-elasmi", argv[i]) == 0) {
            elasmi = 1;
        }
    }

    /* ----------------------- Capture ----------------------- */

    // testing the absence of synchronization
    int sleep_time = 20;

    if(doCapture) {
        std::tie(img, std::ignore) = capture(nbImages, FN_CAP_CAM, 1920, 1080, 30, sleep_time);
    }

    /* ----------------------- Scan 3D ----------------------- */
    Mat lutCam;
    Mat lutProj;

    if(doScan) {
		std::string nameCam = "data/cam1/cam%03d.jpg";
		std::string nameProj = "data/proj1/leopard_2560_1080_32B_%03d.jpg";

        if(roys) {
			// Load images
			int nb = 40;

			std::vector<cv::Mat> imagesCam = util::readImages(nameCam.c_str(), 0, nb - 1, -1.0);
			std::vector<cv::Mat> imagesProj = util::readImages(nameProj.c_str(), 0, nb - 1, -1.0);

            testLeopardSeb(imagesCam, imagesProj);
        } 
		else if(elasmi) {
			nameCam = FN_CAP_CAM;
			nameProj = FN_CAP_PROJ;

			int nb = 60;
			int from = 100;

			std::vector<cv::Mat> imagesCam;

			if (img[0].rows != 0) {
				imagesCam = util::readImagesFromCam(img, from, from + nb - 1);
			}
			else {
				imagesCam = util::readImages(nameCam.c_str(), 0, nb - 1, -1.0);
			}
				
			std::vector<cv::Mat> imagesProj = util::readImages(nameProj.c_str(), 0, nb - 1, -1.0);

            testLeopardChaima(imagesCam, imagesProj, (path + FN_SCAN_LUTC), (path + FN_SCAN_LUTP),
                              (path + FN_SCAN_MIXC), (path + FN_SCAN_MIXP), lutCam, lutProj, doSp);
        }
    } else {
        printf("----- Pas de scan -----\n");

        lutCam  = imread((path + FN_SCAN_LUTC), CV_LOAD_IMAGE_UNCHANGED);
        lutProj = imread((path + FN_SCAN_LUTP), CV_LOAD_IMAGE_UNCHANGED);
    }

    /* ----------------------- Triangulation ----------------------- */
    if(doTriangule) {
        Triangulation tri;

        string pathvide = "";

        // Paths
		tri.setPathT(IDX_TR_MASK, path, FN_TR_MASK);
		tri.setPathT(IDX_TR_DATA, path, FN_TR_DATA);
		tri.setPathT(IDX_TR_PARC, pathvide, FN_TR_PARC);
		tri.setPathT(IDX_TR_PARP, pathvide, FN_TR_PARP);

		tri.triangulate(lutCam, lutProj);
    }

    return 0;
}
