#include <iostream>
#include <string>
#include <iomanip>
#include <sstream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

static void showUsage(const char *av0)
{
    std::cout << av0 << ": Measure video similarity with PSNR and MSSIM."
              << std::endl << std::endl
              << "Usage: " << av0 << " <reference> <test> <trigger> <delay>"
              << std::endl << std::endl
              << "Where: <reference> is a video file against which to"
              << std::endl
              << "                   measure <test>."
              << std::endl
              << "       <test> is a video file similar to <reference>."
              << std::endl
              << "       <trigger> is the PSNR trigger value above which"
              << std::endl
              << "                 PSNR is a useful measure of difference."
              << std::endl
              << "       <delay> is the time to pause between frames."
              << std::endl << std::endl
              << "Example: " << av0 << " ../resources/Megamind.avi \\"
              << std::endl
              << "                     ../resources/Megamind_bugy.avi 35 10"
              << std::endl << std::endl;
}

// Create a new unobscured named window for image.
// Reset windows layout with when reset is not 0.
//
// The 23 term works around how MacOSX decorates windows.
//
static void makeWindow(const char *window, cv::Size size, int reset = 0)
{
    static int across = 1;
    static int count, moveX, moveY, maxY = 0;
    if (reset) {
        across = reset;
        count = moveX = moveY = maxY = 0;
    }
    if (count % across == 0) {
        moveY += maxY + 23;
        maxY = moveX = 0;
    }
    ++count;
    cv::namedWindow(window, cv::WINDOW_AUTOSIZE);
    cv::moveWindow(window, moveX, moveY);
    moveX += size.width;
    maxY = std::max(maxY, size.height);
}

// Return the PSNR between image1 and image1 or 0.0 if below epsilon.
//
static double getPsnr(const cv::Mat &image1, const cv::Mat &image2)
{
    static const int max = std::numeric_limits<uchar>::max();
    static const double maxSquared = max * max;
    static const double epsilon    = 1e-10;
    double result = 0;                  // Return 0 for sse < epsilon.
    cv::Mat intDiff;                    // |image1 - image2|
    cv::Mat floatDiffSquared;           // |image1 - image2|^2
    cv::absdiff(image1, image2, intDiff);
    intDiff.convertTo(floatDiffSquared, CV_32F);
    floatDiffSquared = floatDiffSquared.mul(floatDiffSquared);
    const cv::Scalar sum = cv::sum(floatDiffSquared);
    const double sse = sum.val[0] + sum.val[1] + sum.val[2];
    if (sse > epsilon) {
        const double mse = sse / image1.channels() / image1.total();
        result = 10.0 * log10(maxSquared / mse);
    }
    return result;
}

// Return the MSSIM calculated over image1 and image2.
//
static cv::Scalar getMssim(const cv::Mat &image1, const cv::Mat &image2)
{
    cv::Mat float1;
    cv::Mat float2;
    image1.convertTo(float1, CV_32F);
    image2.convertTo(float2, CV_32F);
    const cv::Mat float2squared = float2.mul(float2); // float2^2
    const cv::Mat float1squared = float1.mul(float1); // float1^2
    const cv::Mat float1x2      = float1.mul(float2); // float1 * float2

    static const cv::Size kernel(11, 11);
    static const double sigmaX = 1.5;
    cv::Mat mu1;
    cv::Mat mu2;
    cv::GaussianBlur(float1, mu1, kernel, sigmaX);
    cv::GaussianBlur(float2, mu2, kernel, sigmaX);
    const cv::Mat mu1squared = mu1.mul(mu1);
    const cv::Mat mu2squared = mu2.mul(mu2);
    const cv::Mat mu1_x_mu2  = mu1.mul(mu2);

    cv::Mat sigma1squared;
    cv::Mat sigma2squared;
    cv::Mat sigma1x2;
    cv::GaussianBlur(float1squared, sigma1squared, kernel, sigmaX);
    cv::GaussianBlur(float2squared, sigma2squared, kernel, sigmaX);
    cv::GaussianBlur(float1x2, sigma1x2, kernel, sigmaX);
    sigma1squared -= mu1squared;
    sigma2squared -= mu2squared;
    sigma1x2 -= mu1_x_mu2;

    // t3 = ((2 * mu1_x_mu2 + C1) * (2 * sigma1x2 + C2))
    // t1 = ((mu1squared + mu2squared + C1)
    //    * (sigma1squared + sigma2squared + C2))
    // ssimMap =  t3 / t1;
    //
    static const double C1 =  6.5025;
    static const double C2 = 58.5225;
    cv::Mat t1 = 2 * mu1_x_mu2 + C1;
    cv::Mat t2 = 2 * sigma1x2  + C2;
    cv::Mat t3 = t1.mul(t2);
    t1 = mu1squared    + mu2squared    + C1;
    t2 = sigma1squared + sigma2squared + C2;
    t1 = t1.mul(t2);
    cv::Mat ssimMap;
    cv::divide(t3, t1, ssimMap);

    // mssim = average of ssim map
    //
    const cv::Scalar result = cv::mean(ssimMap);
    return result;
}


// Just cv::VideoCapture extended for convenience.
//
struct CvVideoCapture: cv::VideoCapture {
    int count(void) {
        return this->get(CV_CAP_PROP_FRAME_COUNT);
    }
    cv::Size size(void) {
        const int w = this->get(CV_CAP_PROP_FRAME_WIDTH);
        const int h = this->get(CV_CAP_PROP_FRAME_HEIGHT);
        const cv::Size result(w, h);
        return result;
    }
    CvVideoCapture(const std::string &fileName): VideoCapture(fileName) {}
};


// Format these nicely on an ostream.
//
#define DECIBEL(PSNR) std::setiosflags(std::ios::fixed)         \
    << std::setw(8) << std::setprecision(3) << (PSNR) << " dB"
#define PERCENT(SSIM) std::setiosflags(std::ios::fixed)                 \
    << std::setw(6) << std::setprecision(2) << (SSIM) * 100 << "%"


// Compare test to reference using PSNR, and if PSNR is less than trigger
// also show MSSIM, with delay ms between frames.
//
static void compareVideos(CvVideoCapture &reference, CvVideoCapture &test,
                          int trigger, int delay)
{
    const cv::Size size = reference.size();
    const int count = std::min(reference.count(), test.count());
    makeWindow("Reference", size, 2);
    makeWindow("Test", size);
    for (int i = 0; i < reference.count(); ++i) {
        std::cout << "Frame " << std::setw(3) << i << ": ";
        cv::Mat rFrame, tFrame; reference >> rFrame; test >> tFrame;
        if (rFrame.empty() || tFrame.empty()) {
            std::cout << "is empty!" << std::endl;
        } else {
            const double psnr = getPsnr(rFrame, tFrame);
            std::cout << "   PSNR:" << DECIBEL(psnr);
            if (psnr > 0.0 && psnr < trigger) {
                const cv::Scalar mssim = getMssim(rFrame, tFrame);
                std::cout << ",   MSSIM:"
                          << "  R" << PERCENT(mssim.val[2])
                          << "  G" << PERCENT(mssim.val[1])
                          << "  B" << PERCENT(mssim.val[0]);
            }
            std::cout << std::endl;
            cv::imshow("Reference", rFrame);
            cv::imshow("Test", tFrame);
            const int c = cv::waitKey(delay);
            if (c != -1) break;
        }
    }
}

int main(int ac, char *av[])
{
    if (ac == 5) {
        std::stringstream s; s << av[3] << std::endl << av[4];
        int trigger = 0, delay = 0; s >> trigger >> delay;
        CvVideoCapture reference(av[1]), test(av[2]);
        const bool ok = trigger && delay
            && reference.isOpened() && test.isOpened()
            && reference.size() == test.size();
        const cv::Size size = reference.size();
        if (ok) {
            std::cout << std::endl << av[0] << ": Press any key to quit."
                      << std::endl << std::endl
                      << reference.count() << " frames (W x H): "
                      << size.width << " x " << size.height
                      << " with PSNR trigger " << trigger
                      << " and delay " << delay << std::endl << std::endl;
            compareVideos(reference, test, trigger, delay);
            return 0;
        }
    }
    showUsage(av[0]);
    return 1;
}
