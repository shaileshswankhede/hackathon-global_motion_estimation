#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main()
{
    std::cout << "Hello World!"  << std::endl;

    std::string img = "lenna.jpg";
    Mat srcImage = imread(img);
    if (!srcImage.data) {
        return 1;
    }
    imshow("srcImage", srcImage);
    waitKey(0);

    return 0;
}