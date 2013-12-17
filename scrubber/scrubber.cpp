#include <opencv2/highgui/highgui.hpp>
#include <iostream>


// Just cv::VideoCapture extended for the convenience of VideoPlayer.
//
struct CvVideoCapture: cv::VideoCapture {

    double getFramesPerSecond() {
        const double fps = this->get(cv::CAP_PROP_FPS);
        return fps ? fps : 30.0;        // for MacBook iSight camera
    }

    int getFourCcCodec() {
        return this->get(cv::CAP_PROP_FOURCC);
    }

    const char *getFourCcCodecString() {
        static int code = 0;
        static char result[5] = "";
        if (code == 0) {
            code = this->getFourCcCodec();
            result[0] = ((code & 0x000000ff) >>  0);
            result[1] = ((code & 0x0000ff00) >>  8);
            result[2] = ((code & 0x00ff0000) >> 16);
            result[3] = ((code & 0xff000000) >> 24);
            result[4] = ""[0];
        }
        return result;
    }

    int getFrameCount() {
        return this->get(cv::CAP_PROP_FRAME_COUNT);
    }

    cv::Size getFrameSize() {
        const int w = this->get(cv::CAP_PROP_FRAME_WIDTH);
        const int h = this->get(cv::CAP_PROP_FRAME_HEIGHT);
        const cv::Size result(w, h);
        return result;
    }

    int getPosition(void) { return this->get(cv::CAP_PROP_POS_FRAMES); }
    void setPosition(int p) { this->set(cv::CAP_PROP_POS_FRAMES, p); }

    CvVideoCapture(const std::string &fileName): VideoCapture(fileName) {}
    CvVideoCapture(int n): VideoCapture(n) {}
    CvVideoCapture(): VideoCapture() {}
};


// Play video from file with title at FPS or by stepping frames using a
// trackbar as a scrub control.
//
class VideoPlayer {

    CvVideoCapture video;
    const char *const title;
    const int msDelay;
    const int frameCount;
    cv::Mat frame;
    int position;
    enum State { RUN, STEP } state;

    // Show the frame at position updating the trackbar as necessary.
    //
    void showFrame(void) {
        video >> frame;
        if (frame.data) {
            position = video.getPosition();
            cv::setTrackbarPos("Position", title, position);
            cv::imshow(title, frame);
        }
    }

    // This is the trackbar callback where p is this VideoPlayer.
    //
    static void track(int position, void *p)
    {
        VideoPlayer *const pV = (VideoPlayer *)p;
        pV->video.setPosition(position);
        pV->state = VideoPlayer::STEP;
        pV->showFrame();
    }

public:

    ~VideoPlayer() { cv::destroyWindow(title); }

    // Show frames at FPS if RUNning or one at a time if STEPping.
    //
    void operator()(void) {
        while (true) {
            showFrame();
            const int wait = state == RUN ? msDelay : 0;
            const char c = cv::waitKey(wait);
            switch (c) {
            case 'q': case 'Q': return;
            case 'r': case 'R': state = RUN;  break;
            case 's': case 'S': state = STEP; break;
            }
        }
    }

    // True if this can play.
    //
    operator bool() const { return video.isOpened(); }

    VideoPlayer(const char *t):
        video(t), msDelay(1000 / video.getFramesPerSecond()),
        frameCount(video.getFrameCount()),
        title(t), position(0), state(STEP)
    {
        if (*this) {
            cv::namedWindow(title, cv::WINDOW_AUTOSIZE);
            cv::createTrackbar("Position", title, &position, frameCount,
                               &VideoPlayer::track, this);
        }
    }
};


int main(int ac, const char *av[])
{
    if (ac == 2) {
        VideoPlayer play(av[1]);
        if (play) {
            std::cout << std::endl
                      << av[0] << ": Press q to quit." << std::endl
                      << av[0] << ": Press r to run video." << std::endl
                      << av[0] << ": Press s to step a frame." << std::endl
                      << av[0] << ": Or drag the Position trackbar."
                      << std::endl;
            play();
            return 0;
        }
    }
    std::cerr << av[0] << ": Show a video with scrubber control." << std::endl
              << std::endl
              << "Usage: " << av[0] << " <video-file>" << std::endl
              << std::endl
              << "Where: <video-file> is a video file." << std::endl
              << std::endl
              << "Example: " << av[0] << " ../resources/Megamind.avi"
              << std::endl << std::endl;
    return 1;
}
