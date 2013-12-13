#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


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
    if (ac == 2) {
        const char *const imageName = av[1];
        const cv::Mat image = cv::imread(imageName, 1);
        if (image.data) {
            cv::Mat grayImage;
            cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
            cv::imwrite("./gray-image.jpg", grayImage);
            cv::namedWindow(imageName, cv::WINDOW_AUTOSIZE);
            cv::moveWindow(imageName, 0, 0);
            cv::imshow(imageName, image);
            cv::namedWindow("Gray Image", cv::WINDOW_AUTOSIZE);
            cv::moveWindow("Gray Image", image.cols, 0);
            cv::imshow("Gray Image", grayImage);
            cv::waitKey(0);
            dumpBunchOfMats();
            return 0;
        }
    }
    std::cerr << av[0] << ": Load and display an image from a file."
              << std::endl << std::endl
              << "Usage: " << av[0] << " <image-file>" << std::endl
              << std::endl
              << "Where: <image-file> is the name of an image file." 
              << std::endl << std::endl
              << "Example: " << av[0] << " ../resources/Twas_Ever_Thus500.jpg"
              << std::endl << std::endl;
    return 1;
}
