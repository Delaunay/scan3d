#include "util.hpp"

using namespace cv;

namespace util{
std::mt19937 &random_engine() {
	std::random_device rd;
	static std::mt19937 engine(rd());
	return engine;
}

double drand48() {
	static std::uniform_real_distribution<> dist(0, 1);
	return dist(random_engine());
}

// On passe le nom des images
std::vector<cv::Mat> readImages(const char *name, int from, int to, double fct) {
    printf("-- reading images %s --\n", name);
    int nb = to - from + 1;
    char buf[300];
    std::vector<cv::Mat> image(nb);
    int w = 0, h = 0;

    for(int i = 0; i < nb; i++) {
        sprintf(buf, name, i + from);
        printf("read %d %s\n", i, buf);

        if(fct > 0) {
            Mat tmp = imread(buf, CV_LOAD_IMAGE_GRAYSCALE);
            resize(tmp, image[i], cvSize(0, 0), 0.5, 0.5);
        } else {
            image[i] = imread(buf, CV_LOAD_IMAGE_GRAYSCALE);
        }

        // printf("loaded %d x %d\n",image[i].cols,image[i].rows);
        if(i == 0) {
            w = image[i].cols;
            h = image[i].rows;
        } else {
            if(w != image[i].cols || h != image[i].rows) {
                printf("Images %d pas de la meme taille!\n", i + from);
                return std::vector<cv::Mat>();
            }
        }
    }
    return image;
}

// On passe les images directement
std::vector<cv::Mat> readImagesFromCam(const std::vector<Mat> &cam, int from, int to) {
    printf("-- reading images from camera --\n");
    int nb = to - from + 1;
    std::vector<cv::Mat> image(nb);
    int w = 0, h = 0;

    for(int i = 0; i < nb; i++) {
        printf("read %d cam_%d \n", i, i + from - 1);
        cvtColor(cam[i + from], image[i], CV_RGB2GRAY);

        if(i == 0) {
            w = image[i].cols;
            h = image[i].rows;
        } else {
            if(w != image[i].cols || h != image[i].rows) {
                printf("Images %d pas de la meme taille!\n", i + from);
                return std::vector<cv::Mat>();
            }
        }
    }
    return image;
}
}