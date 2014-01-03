#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <iostream>


static void showUsage(const char *av0)
{
    static const char bodies[]
        = "../resources/haarcascade_upperbody.xml";
    static const char faces[]
        = "../resources/haarcascade_frontalface_alt.xml";
    static const char eyes[]
        = "../resources/haarcascade_eye_tree_eyeglasses.xml";
    std::cerr << av0 << ": Use Haar cascade classifier to people."
              << std::endl
              << "Usage: " << av0 << " <camera> <faces> <eyes>" << std::endl
              << std::endl
              << "Where: <camera> is an integer camera number." << std::endl
              << "       <bodies> is Haar data for bodies." << std::endl
              << "       <faces>  is Haar data for faces." << std::endl
              << "       <eyes>   is Haar data for eyes." << std::endl
              << std::endl
              << "Example: " << av0 << " 0 " << bodies << " \\ " << std::endl
              << "         " << faces << " \\ " << std::endl
              << "         " << eyes << std::endl << std::endl;
}


// Return an equalized grayscale copy of image.
//
static cv::Mat grayScale(const cv::Mat &image) {
    cv::Mat result;
    cv::cvtColor(image, result, cv::COLOR_RGB2GRAY);
    cv::equalizeHist(result, result);
    return result;
}

// Return regions of interest detected by classifier in gray.
//
static std::vector<cv::Rect> detectCascade(cv::CascadeClassifier &classifier,
                                           const cv::Mat &gray)
{
    static double scaleFactor = 1.1;
    static const int minNeighbors = 2;
    static const cv::Size minSize(30, 30);
    static const cv::Size maxSize;
    std::vector<cv::Rect> result;
    classifier.detectMultiScale(gray, result, scaleFactor, minNeighbors,
                                cv::CASCADE_SCALE_IMAGE, minSize, maxSize);
    return result;
}


// Draw on frame a circle with color at center with radius.
//
static void drawCircle(cv::Mat &frame, const cv::Scalar &color,
                       const cv::Point &center, int radius)
{
    static const int thickness = 4;
    static const int lineKind = 8;
    static const int shift = 0;
    cv::circle(frame, center, radius, color, thickness, lineKind, shift);
}


// Draw rectangle r in blue on frame.
//
static void drawRectangle(cv::Mat &frame, const cv::Rect &r)
{
    static const cv::Scalar blue(255, 0, 0);
    static const int thickness = 4;
    static const int lineKind = 8;
    static const int shift = 0;
    cv::rectangle(frame, r, blue, thickness, lineKind, shift);
}



// Draw red circles around eyes in face.
//
static void drawEyes(cv::Mat &frame, const cv::Rect &body,
                     const cv::Rect &face,
                     const std::vector<cv::Rect> &eyes)
{
    static const cv::Scalar red(0, 0, 255);
    const float fX = body.x + face.x;
    const float fY = body.y + face.y;
    for (size_t j = 0; j < eyes.size(); ++j) {
        const cv::Rect &eye = eyes[j];
        const float eX = fX + eye.x;
        const float eY = fY + eye.y;
        const cv::Point center(eX + eye.width  * 0.5,
                               eY + eye.height * 0.5);
        const int radius = cvRound((eye.width + eye.height) * 0.25);
        drawCircle(frame, red, center, radius);
    }
}


// Draw a green circle around any detected face in body.
//
static void drawFace(cv::Mat &frame, const cv::Rect &body,
                     const cv::Rect &face,
                     const std::vector<cv::Rect> &eyes)
{
    static const cv::Scalar green(0, 255, 0);
    const float fX = body.x + face.x;
    const float fY = body.y + face.y;
    const cv::Point center(fX + face.width * 0.5, fY + face.height * 0.5);
    const int radius = cvRound((face.width + face.height) * 0.25);
    drawCircle(frame, green, center, radius);
    drawEyes(frame, body, face, eyes);
}


// Draw a blue rectangle around any body in the frame.
// Then draw the face within.
//
static void drawBody(cv::Mat &frame, const cv::Rect &body,
                     const std::vector<cv::Rect> &faces,
                     const std::vector<cv::Rect> &eyes)
{
    drawRectangle(frame, body);
    for (size_t i = 0; i < faces.size(); ++i) {
        drawFace(frame, body, faces[i], eyes);
    }
}

// Detect any body in the frame.  Within the body's region of interest
// detect any face, and detect eyes within the face's region of interest.
//
static void displayBody(cv::Mat &frame,
                        cv::CascadeClassifier &bodyHaar,
                        cv::CascadeClassifier &faceHaar,
                        cv::CascadeClassifier &eyesHaar)
{
    const cv::Mat gray = grayScale(frame);
    const std::vector<cv::Rect> bodies = detectCascade(bodyHaar, gray);
    for (size_t i = 0; i < bodies.size(); ++i) {
        const cv::Mat bodyROI = gray(bodies[i]);
        const std::vector<cv::Rect> faces = detectCascade(faceHaar, bodyROI);
        std::vector<cv::Rect> eyes;
        if (!faces.empty()) {
            const cv::Mat faceROI = bodyROI(faces[0]);
            eyes = detectCascade(eyesHaar, faceROI);
        }
        drawBody(frame, bodies[i], faces, eyes);
    }
    cv::imshow("Viola-Jones-Lienhart Classifier", frame);
}


// Just cv::VideoCapture extended for convenience.  The const_cast<>()s
// work around the missing member const on cv::VideoCapture::get().
//
struct CvVideoCapture: cv::VideoCapture {

    double getFramesPerSecond() const {
        CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
        const double fps = p->get(cv::CAP_PROP_FPS);
        return fps ? fps : 30.0;        // for MacBook iSight camera
    }

    int getFourCcCodec() const {
        CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
        return p->get(cv::CAP_PROP_FOURCC);
    }

    std::string getFourCcCodecString() const {
        char result[] = "????";
        CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
        const int code = p->getFourCcCodec();
        result[0] = ((code >>  0) & 0xff);
        result[1] = ((code >>  8) & 0xff);
        result[2] = ((code >> 16) & 0xff);
        result[3] = ((code >> 24) & 0xff);
        result[4] = ""[0];
        return std::string(result);
    }

    int getFrameCount() const {
        CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
        return p->get(cv::CAP_PROP_FRAME_COUNT);
    }

    cv::Size getFrameSize() const {
        CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
        const int w = p->get(cv::CAP_PROP_FRAME_WIDTH);
        const int h = p->get(cv::CAP_PROP_FRAME_HEIGHT);
        const cv::Size result(w, h);
        return result;
    }

    int getPosition(void) const {
        CvVideoCapture *const p = const_cast<CvVideoCapture *>(this);
        return p->get(cv::CAP_PROP_POS_FRAMES);
    }
    void setPosition(int p) { this->set(cv::CAP_PROP_POS_FRAMES, p); }

    CvVideoCapture(const std::string &fileName): VideoCapture(fileName) {}
    CvVideoCapture(int n): VideoCapture(n) {}
    CvVideoCapture(): VideoCapture() {}
};


int main(int ac, const char *av[])
{
    if (ac == 5) {
        int cameraId = 0;
        std::istringstream iss(av[1]); iss >> cameraId;
        cv::CascadeClassifier bodyHaar(av[2]);
        cv::CascadeClassifier faceHaar(av[3]);
        cv::CascadeClassifier eyesHaar(av[4]);
        std::cout << av[0] << ": camera ID " << cameraId << std::endl
                  << av[0] << ": Body data from " << av[2] << std::endl
                  << av[0] << ": Face data from " << av[3] << std::endl
                  << av[0] << ": Eyes data from " << av[4] << std::endl;
        if (!bodyHaar.empty() && !faceHaar.empty() && ! eyesHaar.empty()) {
            CvVideoCapture camera(cameraId);
            std::cout << std::endl << av[0] << ": Press any key to quit."
                      << std::endl << std::endl;
            const int msPerFrame = 1000.0 / camera.getFramesPerSecond();
            while (true) {
                cv::Mat frame; camera >> frame;
                if (!frame.empty()) {
                    displayBody(frame, bodyHaar, faceHaar, eyesHaar);
                }
                const int c = cv::waitKey(msPerFrame);
                if (c != -1) break;
            }
            return 0;
        }
    }
    showUsage(av[0]);
    return 1;
}
