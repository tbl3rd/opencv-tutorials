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
    double framesPerSecond() {
        return this->get(CV_CAP_PROP_FPS);
    }
    int fourCcCodec() {
        return this->get(CV_CAP_PROP_FOURCC);
    }
    const char *fourCcCodecString() {
        static int code = 0;
        static char result[5] = "";
        if (code == 0) {
            code = this->fourCcCodec();
            result[0] = ((code & 0x000000ff) >>  0);
            result[1] = ((code & 0x0000ff00) >>  8);
            result[2] = ((code & 0x00ff0000) >> 16);
            result[3] = ((code & 0xff000000) >> 24);
            result[4] = ""[0];
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
    CvVideoCapture(): VideoCapture() {}
};

// Open ac VideoWriter files like vc named on av into vw.
// Return true unless something goes wrong.
//
static bool openChannelFiles(int ac, char *av[],
                             CvVideoCapture &vc, cv::VideoWriter vw[])
{
    static const bool isColor = true;
    const int codec = vc.fourCcCodec();
    const double fps = vc.framesPerSecond();
    const cv::Size size = vc.frameSize();
    for (int i = 0; i < ac; ++i) {
        vw[i].open(av[i], codec, fps, size, isColor);
    }
    for (int i = 0; i < ac; ++i) if (!vw[i].isOpened()) return false;
    return true;
}

// Separate the count channels of input into output.
//
static void separateChannels(int count, CvVideoCapture &input,
                             cv::VideoWriter output[])
{
    while (true) {
        cv::Mat inFrame;
        input >> inFrame;
        if (inFrame.empty()) break;
        for (int color = 0; color < count; ++color) {
            std::vector<cv::Mat> channel;
            cv::split(inFrame, channel);
            const cv::Mat black
                = cv::Mat::zeros(channel[0].size(), channel[0].type());
            for (int i = 0; i < channel.size(); ++i) {
                if (i != color) channel[i] = black;
            }
            cv::Mat outFrame;
            cv::merge(channel, outFrame);
            output[color] << outFrame;
        }
    }
}

// Play the ac VideoCapture files named in av[].
//
struct VideoShow { const char *name; CvVideoCapture vc; cv::Mat frame; };
static void playVideo(int ac, char *av[])
{
    bool ok = true;
    std::vector<VideoShow> video(ac);
    for (int i = 0; ok && i < ac; ++i) {
        video[i].name = av[i];
        video[i].vc.open(video[i].name);
        ok = ok && video[i].vc.isOpened();
    }
    if (ok) {
        for (int i = 0; ok && i < ac; ++i) {
            makeWindow(video[i].name, video[i].vc.frameSize(), i == 0? 2: 0);
        }
        const int msFrameDelay = 1.0 / video[0].vc.framesPerSecond() * 1000;
        while (ok) {
            for (int i = 0; ok && i < ac; ++i) {
                video[i].vc >> video[i].frame;
                ok = !video[i].frame.empty();
                if (ok) cv::imshow(video[i].name, video[i].frame);
            }
            if (ok) {
                const int c = cv::waitKey(msFrameDelay);
                ok = c == -1;
            }
        }
    }
}

int main(int ac, char *av[])
{
    enum { BLUE, GREEN, RED, COUNT };
    if (ac == 2 + COUNT) {
        cv::VideoWriter output[COUNT];
        CvVideoCapture input(av[1]);
        bool ok = input.isOpened();
        if (ok) ok = openChannelFiles(ac - 2, av + 2, input, output);
        if (ok) {
            separateChannels(COUNT, input, output);
            std::cout << std::endl << av[0] << ": Press any key to quit."
                      << std::endl << std::endl
                      << input.frameCount() << " frames ("
                      << input.frameSize().width << " x "
                      << input.frameSize().height
                      << ") with codec " << input.fourCcCodecString()
                      << " at " << input.framesPerSecond()
                      << " frames/second." << std::endl << std::endl;
            playVideo(ac - 1, av + 1);
            return 0;
        }
    }
    showUsage(av[0]);
    return 1;
}
