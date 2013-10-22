#include <opencv/cv.h>
#include <opencv/highgui.h>

static void dumpBunchOfMats(void)
{
    cv::Mat zeros = cv::Mat::zeros(4, 4, CV_8UC(2));
    std::cout << "zeros = " << std::endl << " " << zeros << std::endl;
    cv::Mat eyes = cv::Mat::eye(4, 4, CV_8UC(2));
    std::cout << "eyes = " << std::endl << " " << eyes << std::endl;
    cv::Mat doubles = (cv::Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
    std::cout << "doubles = " << std::endl << " " << doubles << std::endl;
    cv::Mat randoms(3, 2, CV_8UC3);
    cv::randu(randoms, cv::Scalar::all(0), cv::Scalar::all(255));
    std::cout << "randoms = " << std::endl << " " << randoms << std::endl;
}

int main (int ac, char *av[])
{
    const char *const imageName = av[1];
    const cv::Mat image = cv::imread(imageName, 1);
    if (ac != 2 || !image.data) {
        std::cout << "No image data" << std::endl;
        return 1;
    }
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, CV_BGR2GRAY);
    cv::imwrite("./gray-image.jpg", grayImage);
    cv::namedWindow(imageName, CV_WINDOW_AUTOSIZE);
    cv::namedWindow("Gray Image", CV_WINDOW_AUTOSIZE);
    cv::imshow(imageName, image);
    cv::imshow("Gray Image", grayImage);
    cv::waitKey(0);
    dumpBunchOfMats();
    return 0;
}
