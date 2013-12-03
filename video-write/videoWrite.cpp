#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

static void showUsage(const char *av0)
{
    std::cout << av0 << ": Extract, write, and display video color channels."
              << std::endl << std::endl
              << "Usage: " << av0 << " <input> <r-out> <g-out> <b-out>"
              << std::endl << std::endl
              << "Where: <input> is a color video file." << std::endl
              << "       <b-out> is where to write the blue channel."
              << std::endl
              << "       <g-out> is where to write the green channel."
              << std::endl
              << "       <r-out> is where to write the red channel."
              << std::endl << std::endl
              << "Example: " << av0 << " ../resources/Megamind.avi"
              << " red.avi green.avi blue.avi"
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

// Just cv::VideoCapture extended for convenience.
//
struct CvVideoCapture: cv::VideoCapture {
    int framesPerSecond() {
        return this->get(CV_CAP_PROP_FPS);
    }
    int codec() {
        return this->get(CV_CAP_PROP_FOURCC);
    }
    const char *codecString() {
        static int code = 0;
        static char result[5] = "";
        if (code == 0) {
            code = this->codec();
            result[0] = ((code & 0x000000ff) >>  0);
            result[1] = ((code & 0x0000ff00) >>  8);
            result[2] = ((code & 0x00ff0000) >> 16);
            result[3] = ((code & 0xff000000) >> 24);
            result[4] = 0;
        }
        return result;
    }
    int frameCount() {
        return this->get(CV_CAP_PROP_FRAME_COUNT);
    }
    cv::Size frameSize() {
        const int w = this->get(CV_CAP_PROP_FRAME_WIDTH);
        const int h = this->get(CV_CAP_PROP_FRAME_HEIGHT);
        const cv::Size result(w, h);
        return result;
    }
    CvVideoCapture(const std::string &fileName): VideoCapture(fileName) {}
};

int main(int ac, char *av[])
{
    enum { BLUE, GREEN, RED, COUNT };
    if (ac == 5) {
        cv::VideoWriter out[COUNT];
        CvVideoCapture input(av[1]);
        bool ok = input.isOpened();
        if (ok) {
            const int codec = input.codec();
            const int fps = input.framesPerSecond();
            const cv::Size frameSize = input.frameSize();
            for (int i = 0; i < COUNT; ++i) {
                static const bool isColor = true;
                out[i].open(av[i + 2], codec, fps, frameSize, isColor);
            }
            for (int i = 0; ok && i < COUNT; ++i) ok = out[i].isOpened();
        }
        while (ok) {
            cv::Mat inFrame;
            input >> inFrame;
            if (inFrame.empty()) break;
            for (int color = BLUE; color < COUNT; ++color) {
                std::vector<cv::Mat> channel;
                cv::split(inFrame, channel);
                const cv::Mat black
                    = cv::Mat::zeros(channel[0].size(), channel[0].type());
                for (int i = 0; i < channel.size(); ++i) {
                    if (i != color) channel[i] = black;
                }
                cv::Mat outFrame;
                cv::merge(channel, outFrame);
                out[color] << outFrame;
            }
        }
        if (ok) {
            std::cout << std::endl << av[0] << ": Press any key to quit."
                      << std::endl << std::endl
                      << input.frameCount() << " frames (W x H): "
                      << input.frameSize().width << " x "
                      << input.frameSize().height
                      << std::endl << std::endl;
            return 0;
        }
    }
    showUsage(av[0]);
    return 1;
}
