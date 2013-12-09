#include <iomanip>
#include <iostream>
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>


static void showUsage(const char *av0)
{
    std::cerr << av0 << ": Demonstrate FLANN-based feature matching."
              << std::endl << std::endl
              << "Usage: " << av0 << " <object> <scene>" << std::endl
              << std::endl
              << "Where: <object> and <scene> are image files." << std::endl
              << "       <object> has features present in <scene>." << std::endl
              << "       <scene> is where to search for features" << std::endl
              << "               from the <object> image." << std::endl
              << std::endl
              << "Example: " << av0 << " ../resources/box.png"
              << " ../resources/box_in_scene.png" << std::endl
              << std::endl;
}

// Features in a object image matched to a scene image.
//
typedef std::vector<cv::DMatch> Matches;

// The keypoints and descriptors for features in object or scene image i.
//
struct Features {
    const cv::Mat image;
    std::vector<cv::KeyPoint> keyPoints;
    cv::Mat descriptors;
    Features(const cv::Mat &i): image(i) {}
};

// Return matches of object in scene.
//
static Matches matchFeatures(Features &object, Features &scene)
{
    static const int minHessian = 400;
    cv::SurfFeatureDetector detector(minHessian);
    detector.detect(object.image, object.keyPoints);
    detector.detect(scene.image, scene.keyPoints);
    cv::SurfDescriptorExtractor extractor;
    extractor.compute(object.image, object.keyPoints, object.descriptors);
    extractor.compute(scene.image, scene.keyPoints, scene.descriptors);
    cv::FlannBasedMatcher matcher;
    Matches result;
    matcher.match(object.descriptors, scene.descriptors, result);
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
    std::cout << "Minimum distance: " << minDist << std::endl
              << "Maximum distance: " << maxDist << std::endl;
    const double threshold = std::max(2 * minDist, epsilon);
    Matches result;
    for (int i = 0; i < matches.size(); ++i) {
        if (matches[i].distance <= threshold) result.push_back(matches[i]);
    }
    return result;
}

// Return image with matches drawn from object to scene in random colors.
//
static cv::Mat drawMatches(Features &object, Features &scene,
                           const Matches &matches)
{
    static const cv::Scalar color = cv::Scalar::all(-1);
    static const std::vector<char> noMask;
    static const int flags
        = cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
        | cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
    cv::Mat result;
    cv::drawMatches(object.image, object.keyPoints,
                    scene.image, scene.keyPoints,
                    matches, result, color, color, noMask, flags);
    return result;
}

int main(int ac, char *av[])
{
    if (ac == 3) {
        Features  object(cv::imread(av[1], cv::IMREAD_GRAYSCALE));
        Features scene(cv::imread(av[2], cv::IMREAD_GRAYSCALE));
        if (object.image.data && scene.image.data) {
            std::cout << std::endl << av[0] << ": Press any key to quit."
                      << std::endl << std::endl;
            const Matches matches = matchFeatures(object, scene);
            const Matches good = goodMatches(matches);
            const cv::Mat image = drawMatches(object, scene, good);
            const int count = good.size();
            std::stringstream ss; ss << count << " Good Matches" << std::ends;
            cv::imshow(ss.str(), image);
            std::cout << std::endl;
            for (int i = 0; i < count; ++i) {
                std::cout << "Match"  << std::setw(2) << i << ": "
                          << "Object:"  << std::setw(4) << good[i].queryIdx
                          << ", "
                          << "Scene:" << std::setw(4) << good[i].trainIdx
                          << std::endl;
            }
            cv::waitKey(0);
            return 0;
        }
    }
    showUsage(av[0]);
    return 1;
}
