#include <iomanip>
#include <iostream>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>


static void showUsage(const char *av0)
{
    std::cerr << av0 << ": Demonstrate FLANN-based feature matching."
              << std::endl << std::endl
              << "Usage: " << av0 << " <goal> <scene>" << std::endl
              << std::endl
              << "Where: <goal> and <scene> are image files." << std::endl
              << "       <goal> has features present in <scene>." << std::endl
              << "       <scene> is where to search for features" << std::endl
              << "               from the <goal> image." << std::endl
              << std::endl
              << "Example: " << av0 << " ../resources/box.png"
              << " ../resources/box_in_scene.png" << std::endl
              << std::endl;
}

typedef std::vector<cv::DMatch> Matches;

struct Features {
    const cv::Mat image;
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors;
    Features(const cv::Mat &i): image(i) {}
};

// Return matches of goal in scene.
//
static Matches matchFeatures(Features &goal, Features &scene)
{
    static const int minHessian = 400;
    cv::SurfFeatureDetector detector(minHessian);
    detector.detect(goal.image, goal.keyPoints);
    detector.detect(scene.image, scene.keyPoints);
    cv::SurfDescriptorExtractor extractor;
    extractor.compute(goal.image, goal.keyPoints, goal.descriptors);
    extractor.compute(scene.image, scene.keyPoints, scene.descriptors);
    cv::FlannBasedMatcher matcher;
    Matches result;
    matcher.match(goal.descriptors, scene.descriptors, result);
    return result;
}

// Return only good matches in matches.  A good match has distance less
// than twice the minimum distance (or epsilon).
//
// Another option for this filter is cv::radiusMatch().
//
static Matches goodMatches(const Matches &matches)
{
    static const double epsilon = 0.02;
    double minDist = 100.0, maxDist = 0.0;
    for (int i = 0; i < matches.size(); ++i) {
        const double dist = matches[i].distance;
        if (dist < minDist) minDist = dist;
        if (dist > maxDist) maxDist = dist;
    }
    std::cout << "Minimum distance: " << minDist << std::endl;
    std::cout << "Maximum distance: " << maxDist << std::endl;
    Matches result;
    for (int i = 0; i < matches.size(); ++i) {
        const double threshold = std::max(2 * minDist, epsilon);
        if (matches[i].distance <= threshold) {
            result.push_back(matches[i]);
        }
    }
    return result;
}

// Return image with matches drawn from goal to scene in random colors.
//
static cv::Mat drawMatches(Features &goal, Features &scene,
                           const Matches &matches)
{
    static const cv::Scalar color = cv::Scalar::all(-1);
    static const std::vector<char> noMask;
    static const int noSingles = cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS;
    cv::Mat result;
    cv::drawMatches(goal.image, goal.keyPoints, scene.image, scene.keyPoints,
                    matches, result, color, color, noMask, noSingles);
    return result;
}

int main(int ac, char *av[])
{
    if (ac == 3) {
        Features  goal(cv::imread(av[1], cv::IMREAD_GRAYSCALE));
        Features scene(cv::imread(av[2], cv::IMREAD_GRAYSCALE));
        if (goal.image.data && scene.image.data) {
            std::cout << std::endl << av[0] << ": Press any key to quit."
                      << std::endl << std::endl;
            const Matches matches = matchFeatures(goal, scene);
            const Matches good = goodMatches(matches);
            const cv::Mat image = drawMatches(goal, scene, good);
            const int count = good.size();
            std::stringstream ss; ss << count << " Good Matches" << std::ends;
            cv::imshow(ss.str(), image);
            std::cout << std::endl;
            for (int i = 0; i < count; ++i) {
                std::cout << "Match"       << std::setw(2) << i << ": "
                          << "Keypoint 1:" << std::setw(4) << good[i].queryIdx
                          << ", "
                          << "Keypoint 2:" << std::setw(4) << good[i].trainIdx
                          << std::endl;
            }
            cv::waitKey(0);
            return 0;
        }
    }
    showUsage(av[0]);
    return 1;
}
