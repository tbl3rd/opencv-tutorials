#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>


// Return an equalized grayscale copy of image.
//
static cv::Mat grayScale(const cv::Mat &image) {
    cv::Mat result;
    cv::cvtColor(image, result, cv::COLOR_RGB2GRAY);
    cv::equalizeHist(result, result);
    return result;
}

static std::vector<cv::Rect> detectCascade(cv::CascadeClassifier &cc,
                                           const cv::Mat &gray)
{
    static double scaleFactor = 1.1;
    static const int minNeighbors = 2;
    static const cv::Size minSize(30, 30);
    static const cv::Size maxSize;
    std::vector<cv::Rect> result;
    cc.detectMultiScale(gray, result, scaleFactor, minNeighbors,
                        CV_HAAR_SCALE_IMAGE, minSize, maxSize);
    return result;
}

static void drawFace(cv::Mat &frame, const cv::Rect &face,
                     const std::vector<cv::Rect> &eyes)
{
    static const cv::Scalar faceColor(255, 0, 255);
    static const cv::Scalar eyesColor(255, 0,   0);
    static const double angle = 0.0;
    static const double beginAngle = 0.0;
    static const double endAngle = 360.0;
    static const int thickness = 4;
    static const int lineKind = 8;
    static const int shift = 0;
    const cv::Size axes(face.width * 0.5, face.height * 0.5);
    const cv::Point center(face.x + axes.width, face.y + axes.height);
    cv::ellipse(frame, center, axes, angle, beginAngle, endAngle, faceColor,
                thickness, lineKind, shift);
    for (size_t j = 0; j < eyes.size(); ++j) {
        const cv::Rect &eye = eyes[j];
        const cv::Point center(face.x + eye.x + eye.width   * 0.5,
                               face.y + eye.y + eye.height  * 0.5);
        const int radius = cvRound((eye.width + eye.height) * 0.25);
        cv::circle(frame, center, radius, eyesColor,
                   thickness, lineKind, shift);
    }

}

static void detectAndDisplayFace(cv::Mat &frame,
                                 cv::CascadeClassifier &faceCascade,
                                 cv::CascadeClassifier &eyesCascade)

{
    const cv::Mat gray = grayScale(frame);
    const std::vector<cv::Rect> faces = detectCascade(faceCascade, gray);
    for (size_t i = 0; i < faces.size(); ++i) {
        const cv::Mat faceROI = gray(faces[i]);
        const std::vector<cv::Rect> eyes = detectCascade(eyesCascade, faceROI);
        drawFace(frame, faces[i], eyes);
    }
    cv::imshow("Capture - Face detection", frame);
}

int main(int ac, const char *av[])
{
    cv::CascadeClassifier faceCascade("haarcascade_frontalface_alt.xml");
    cv::CascadeClassifier eyesCascade("haarcascade_eye_tree_eyeglasses.xml");
    if (!faceCascade.empty() && ! eyesCascade.empty()) {
        CvCapture *const capture = cvCaptureFromCAM(-1);
        if (capture) {
            while (true) {
                cv::Mat frame = cvQueryFrame(capture);
                if (frame.empty())  {
                    std::cout << "No frame." << std::endl;
                } else {
                    detectAndDisplayFace(frame, faceCascade, eyesCascade);
                }
                const int c = cv::waitKey(10);
                if (c != -1) break;
            }
        }
        return 0;
    }
    std::cerr << av[0] << ": Detect faces and eyes in live video." << std::endl
              << "Usage: " << av[0] << std::endl;
    return 1;
}
