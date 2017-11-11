﻿#include <QCoreApplication>
#include <opencv2/opencv.hpp>
#include <sys/time.h>
#include <unistd.h>


#include <leopard.hpp>
#include <triangulation.hpp>
#include <paths.hpp>



using namespace cv;
using namespace std;



double horloge() {
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return( (double) tv.tv_sec + tv.tv_usec / 1000000.0);
}


void testLeopardSeb() {
    printf("----- test leopard seb -----\n");


    printf("sizeof char %d\n",(int)sizeof(char));
    printf("sizeof short %d\n",(int)sizeof(short));
    printf("sizeof int %d\n",(int)sizeof(int));
    printf("sizeof long %d\n",(int)sizeof(long));
    printf("sizeof long long %d\n",(int)sizeof(long long));

    leopard *L=new leopard();
    /// lire des images
    int nb=40;
    Mat *imagesCam;
    imagesCam=L->readImages((char *)"data/cam1/cam%03d.jpg",0,nb-1, -1.0);
    L->computeMask(1,imagesCam,nb,1.45,5.0,1,0,0);
    //L->computeCodes(1,LEOPARD_SIMPLE,imagesCam);
    L->computeCodes(1,LEOPARD_QUADRATIC,imagesCam);
    delete[] imagesCam;

    Mat *imagesProj;
    imagesProj=L->readImages((char *)"data/proj1/leopard_2560_1080_32B_%03d.jpg",0,nb-1, -1.0);
    L->computeMask(0,imagesProj,nb,1.45,5.0,1,0,0);
    //L->computeCodes(0,LEOPARD_SIMPLE,imagesProj);
    L->computeCodes(0,LEOPARD_QUADRATIC,imagesProj);
    delete[] imagesProj;

    // quelques stats
    L->statsCodes(1);
    L->statsCodes(0);

    L->prepareMatch();
    //L->forceBrute();
    for(int i=0;i<20;i++) {
        L->doLsh(0);
        //L->doHeuristique();
    }

    cv::Mat lutCam;
    cv::Mat lutProj;
    L->makeLUT(lutCam,1);
    L->makeLUT(lutProj,0);

    imwrite("lutcam.png",lutCam);
    imwrite("lutproj.png",lutProj);


    printf("test\n");
    delete L;
    printf("----- done -----\n");
}


void testLeopardChaima(string nameCam, string nameProj, string namelutC, string namelutP,
                       Mat *imgCam, Mat &lutCam, Mat &lutProj, int sp) {
    printf("----- test leopard chaima -----\n");


    printf("sizeof char %d\n",(int)sizeof(char));
    printf("sizeof short %d\n",(int)sizeof(short));
    printf("sizeof int %d\n",(int)sizeof(int));
    printf("sizeof long %d\n",(int)sizeof(long));
    printf("sizeof long long %d\n",(int)sizeof(long long));

    double timeS = horloge();

    leopard *L=new leopard();
    int nb = 60;
    int from = 20;
    //Camera: Images / Code simple
    Mat *imagesCam;
    if(imgCam->rows != 0) {
        imagesCam = L->readImages2(imgCam, from, from+nb-1);
    }
    else {
        imagesCam = L->readImages((char *) nameCam.c_str(), from, from+nb-1, -1.0);
    }
    L->computeMask(1,imagesCam,nb,0.45,5.0,1,0,0);
    L->computeCodes(1,LEOPARD_SIMPLE,imagesCam);

    //Projecteur: Images / Code simple
    Mat *imagesProj;
    imagesProj=L->readImages((char *) nameProj.c_str(), 0, nb-1, -1.0);
    L->computeMask(0,imagesProj,nb,1,5.0,1,0,0);
    L->computeCodes(0,LEOPARD_SIMPLE,imagesProj);


    //Cherche la premiere image de la séquence camera
    int posR = 0;
    L->prepareMatch();
    posR = L->doShiftCodes();

    Mat *imagesCamDecal = new Mat[nb];
    for(int i=0; i<nb; i++)
        imagesCamDecal[i] = imagesCam[(i+posR)%nb];


    //Cherche le mix des images du projecteur
    Mat *imagesProjMix=new Mat[nb];
    int sumCostS=0, sumCostP=0;

    //Match avec l'image suivante
    L->prepareMatch();
    for(int i=0; i<nb-1; i++)
        imagesProjMix[i] = imagesProj[i]*0.5 + imagesProj[i+1]*0.5;
    imagesProjMix[nb-1] = imagesProj[nb-1];

    //QUAD
    L->computeCodes(1,LEOPARD_QUADRATIC,imagesCamDecal);
    L->computeCodes(0,LEOPARD_QUADRATIC,imagesProjMix);

    for(int j=0; j<10; j++)
        L->doLsh(0);

    sumCostS = L->sumCost();

    //Match avec la précédente
    L->prepareMatch();
    for(int i=nb-1; i>0; i--)
        imagesProjMix[i] = imagesProj[i]*0.5 + imagesProj[i-1]*0.5;
    imagesProjMix[0] = imagesProj[0];

    //QUAD
    L->computeCodes(1,LEOPARD_QUADRATIC,imagesCamDecal);
    L->computeCodes(0,LEOPARD_QUADRATIC,imagesProjMix);

    for(int j=0; j<10; j++)
        L->doLsh(0);

    sumCostP = L->sumCost();

    printf("\n sumSuivante = %d, sumPrécédente = %d \n",sumCostS, sumCostP);

    //Choix du mix
    L->prepareMatch();
    if(sumCostS < sumCostP) {
        printf("\n match avec la suivante ! \n");
        for(double fct=0; fct<=1; fct+=0.3) {
            printf("\n\n---------------------- facteur = %.2f ----------------------\n\n", fct);

            for(int i=0; i<nb-1; i++)
                imagesProjMix[i] = imagesProj[i]*(1-fct) + imagesProj[i+1]*fct;
            imagesProjMix[nb-1] = imagesProj[nb-1];

            //QUAD
            L->computeCodes(1,LEOPARD_QUADRATIC,imagesCamDecal);
            L->computeCodes(0,LEOPARD_QUADRATIC,imagesProjMix);

            //TEST: pas de cumul
            //L->prepareMatch();
            for(int j=0; j<10; j++)
                L->doLsh(sp);
        }
    }
    else {
        printf("\n match avec la précédente ! \n");
        for(double fct=0; fct<=1; fct+=0.3) {
            printf("\n\n---------------------- facteur = %.2f ----------------------\n\n", fct);

            for(int i=nb-1; i>0; i--)
                imagesProjMix[i] = imagesProj[i]*(1-fct) + imagesProj[i-1]*fct;
            imagesProjMix[0] = imagesProj[0];

            //QUAD
            L->computeCodes(1,LEOPARD_QUADRATIC,imagesCamDecal);
            L->computeCodes(0,LEOPARD_QUADRATIC,imagesProjMix);

            //TEST: pas de cumul
            //L->prepareMatch();
            for(int j=0; j<10; j++)
                L->doLsh(sp);
        }
    }

    //L->forceBrute();

    L->makeLUT(lutCam,1);
    L->makeLUT(lutProj,0);
    imwrite(namelutC, lutCam);
    imwrite(namelutP, lutProj);


    double timeE = horloge();
    printf("\n Time = %f \n", timeE-timeS);

    delete[] imagesCam;
    delete[] imagesCamDecal;
    delete[] imagesProj;
    delete[] imagesProjMix;
    delete L;
    printf("----- leopard done -----\n");
}


int main(int argc, char *argv[]) {

    int nbImages = 300;
    Mat img[nbImages];

    string nameCam, nameProj;
    Mat lutCam;
    Mat lutProj;

    nameCam  = FN_CAP_CAM;
    nameProj = FN_CAP_PROJ;



    /* ----------------------- Capture ----------------------- */
    if(CAPTURE) {

        printf("----- Capture -----\n");

        VideoCapture cap(1);
        cap.set(CV_CAP_PROP_FRAME_WIDTH,1920);
        cap.set(CV_CAP_PROP_FRAME_HEIGHT,1080);
        cap.set(CV_CAP_PROP_FPS,30);

        for(int i = 0; i < 30; i++)
            cap >> img[0]; //Démarrer la caméra

        if(!cap.isOpened()) {
            cout << "Camera error" << endl;
            return -1;
        }
        else {
            cout << "Camera ready" << endl;
        }

        srand(time(NULL));
        int random = rand() % 500000;
        cout << "random : " << random << endl;
        usleep(random);

        double timeS = horloge();
        for(int i = 0; i < nbImages; i++) {
            cap >> img[i];
            //resize(img[i], resized[i], Size(640,480)); //(683,384)
        }
        double timeE = horloge();
        double xs, ys;
        xs = cap.get(CV_CAP_PROP_FRAME_WIDTH);
        ys = cap.get(CV_CAP_PROP_FRAME_HEIGHT);
        cout << "x = " << xs << "   y = " << ys << endl;

        cout << "Time : " << (timeE - timeS) / nbImages << endl;
        cout << "Time (" << nbImages << " images): " << (timeE - timeS) << endl;
        waitKey(0);

        namedWindow("Display Image", 1);
        for(int i = 0; i < nbImages; i++) {
            imwrite( format(nameCam.c_str(), i), img[i] );
            imshow("Display Image", img[i]);
            waitKey(30);
        }

        printf("----- Capture done -----\n");
        printf("\n\n");
    }

    /* ----------------------- Scan 3D ----------------------- */
    if(SCAN) {
        char *user=getenv("USER");
        printf("Usager %s\n",user);
        printf("\n\n");


        // options
        for(int i=1;i<argc;i++) {
            if( strcmp("-h",argv[i])==0 ) {
                printf("Usage: %s -h\n",argv[0]);
                exit(0);
            }
        }

        if( strcmp(user,"roys")==0 ) {
            testLeopardSeb();
        }else if( strcmp(user,"chaima")==0 ) {
            testLeopardChaima(nameCam, nameProj, FN_SCAN_LUTC, FN_SCAN_LUTP, img, lutCam, lutProj, 1);
        }
    }
    else {
        printf("----- Pas de scan -----\n");

        printf("erreur 1 \n");
        lutCam  = imread(FN_SCAN_LUTC , CV_LOAD_IMAGE_UNCHANGED);
        lutProj =  imread(FN_SCAN_LUTP, CV_LOAD_IMAGE_UNCHANGED);
    }

    /* ----------------------- Triangulation ----------------------- */
    if(TRIANGULE) {
        triangulate(lutCam, lutProj);
    }

    return 0;
}
