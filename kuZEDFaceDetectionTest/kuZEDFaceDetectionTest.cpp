#include <chrono>
#include <iostream>

#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>

#define ResizeScale		  1
#define SearchRegionScale 0.2

cv::Mat slMat2cvMat(sl::Mat &input) {
	// Mapping between MAT_TYPE and CV_TYPE
	int cv_type = -1;
	switch (input.getDataType()) {
	case sl::MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
	case sl::MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
	case sl::MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
	case sl::MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
	case sl::MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
	case sl::MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
	case sl::MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
	case sl::MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
	default: break;
	}

	// Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
	// cv::Mat and sl::Mat will share a single memory structure
	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(sl::MEM_CPU));
}

void main()
{
	sl::Camera				zedCam;
	sl::InitParameters		initParams;

	initParams.camera_resolution = sl::RESOLUTION_HD720;
	initParams.depth_mode = sl::DEPTH_MODE_PERFORMANCE;
	initParams.coordinate_units = sl::UNIT_MILLIMETER;

	sl::ERROR_CODE	err = zedCam.open(initParams);

	sl::RuntimeParameters	runtimeParams;
	sl::Resolution			imageSize = zedCam.getResolution();
	int		newWidth = imageSize.width / 2;
	int		newHeight = imageSize.height / 2;

	std::cout << "Resolution: (" << imageSize.width << " x " << imageSize.height << ")" << std::endl;

	sl::Mat colorImgZED(imageSize.width, imageSize.height, sl::MAT_TYPE_8U_C4);
	cv::Mat	colorImgCV = slMat2cvMat(colorImgZED);
	cv::Mat colorImgCVQuarter;
	cv::Mat grayImgCVQuarter;

	std::chrono::high_resolution_clock::time_point timeStamp[2];

	timeStamp[0] = std::chrono::high_resolution_clock::now();

	int frameCnt = 0;
	float cummulativeFPS = 0;
	float averageFPS;

	cv::CascadeClassifier faceCascadeHaar;
	faceCascadeHaar.load("Models/haarcascade_frontalface_alt.xml");

	while (true)
	{
		if (zedCam.grab() == sl::SUCCESS)
		{
			timeStamp[1] = std::chrono::high_resolution_clock::now();
			auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(timeStamp[1] - timeStamp[0]);
			timeStamp[0] = timeStamp[1];
			std::cout << "Frame time: " << interval.count() << "ms, FPS: " << (float)1000 / (float)interval.count() << std::endl;

			zedCam.retrieveImage(colorImgZED, sl::VIEW_LEFT, sl::MEM_CPU);
			
			float scale = (float)1 / (float)ResizeScale;

			cv::resize(colorImgCV, colorImgCVQuarter, cv::Size(scale * colorImgCV.cols, scale * colorImgCV.rows));
			cv::Mat colorImgCVQuarterBGR;
			cv::cvtColor(colorImgCVQuarter, grayImgCVQuarter, CV_BGRA2GRAY);
			//cv::equalizeHist(grayImgCVQuarter, grayImgCVQuarter);

			std::vector<cv::Rect> detectedFaces;
			faceCascadeHaar.detectMultiScale(grayImgCVQuarter, detectedFaces, 1.1, 1, 0, cv::Size(300, 300));
			
			if (detectedFaces.size() != 0)
			{
				int largestAreaIdx = 0;
				int largestArea = 0;

				for (int i = 0; i < detectedFaces.size(); i++)
				{
					if (detectedFaces.size() >= 1)
					{
						for (int i = 0; i < detectedFaces.size(); i++)
						{
							int area = detectedFaces[i].area();
							if (area > largestArea)
							{
								largestAreaIdx = i;
								largestArea = area;
							}
						}
					}
				}

				cv::rectangle(colorImgCV, detectedFaces[largestAreaIdx], cv::Scalar(255, 0, 255));

			}

			frameCnt++;
			cummulativeFPS += ((float)1000 / (float)interval.count());
			averageFPS = cummulativeFPS / frameCnt;

			std::cout << "Average FPS: " << averageFPS << std::endl;

			cv::imshow("Image", colorImgCV);
			cv::imshow("QuarterImage", grayImgCVQuarter);
			cv::waitKey(1);
		}
	}

	system("pause");
}