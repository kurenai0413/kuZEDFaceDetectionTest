#include <chrono>
#include <iostream>

#include <sl/Camera.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>

#define ResizeScale		  4
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

	// 原來這樣cv::mat指標會直接指向sl::mat的資料
	sl::Mat colorImgZED(imageSize.width, imageSize.height, sl::MAT_TYPE_8U_C4);
	cv::Mat	colorImgCV = slMat2cvMat(colorImgZED);
	cv::Mat colorImgCVQuarter;
	cv::Mat grayImgCVQuarter;

	std::vector<cv::Rect>	faces;
	std::chrono::high_resolution_clock::time_point timeStamp[2];

	timeStamp[0] = std::chrono::high_resolution_clock::now();

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	dlib::deserialize("shape_predictor_5_face_landmarks.dat") >> pose_model;

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
			//cv::cvtColor(colorImgCVQuarter, colorImgCVQuarterBGR, CV_BGRA2BGR);
			//dlib::cv_image<dlib::bgr_pixel> imgQuarter_dlib(colorImgCVQuarterBGR);
			cv::cvtColor(colorImgCVQuarter, grayImgCVQuarter, CV_BGRA2GRAY);
			dlib::cv_image<uchar> imgQuarter_dlib(grayImgCVQuarter);

			// Detect faces 
			std::vector<dlib::rectangle> faces = detector(imgQuarter_dlib);

			if (faces.size() != 0)
			{
				cv::Point2f tlCorner = cv::Point2f(ResizeScale * (float)faces[0].left(), ResizeScale * (float)faces[0].top());
				cv::Point2f brCorner = cv::Point2f(ResizeScale * (float)faces[0].right(), ResizeScale * (float)faces[0].bottom());

				int originalFaceWidth  = (int)abs(faces[0].left() - faces[0].right());
				int originalFaceHeight = (int)abs(faces[0].top() - faces[0].bottom());

				float l = faces[0].left() - SearchRegionScale * originalFaceWidth;
				float t = faces[0].top() - SearchRegionScale * originalFaceHeight;
				float r = faces[0].right() + SearchRegionScale * originalFaceWidth;
				float b = faces[0].bottom() + SearchRegionScale * originalFaceHeight;

				dlib::rectangle searchROI((long)l, (long)t, (long)r, (long)b);

				cv::Point2f searchROITlCorner = cv::Point2f(ResizeScale * (float)searchROI.left(), ResizeScale * (float)searchROI.top());
				cv::Point2f searchROIBrCorner = cv::Point2f(ResizeScale * (float)searchROI.right(), ResizeScale * (float)searchROI.bottom());
				
				cv::rectangle(colorImgCV, tlCorner, brCorner, CV_RGB(0, 0, 255), 1, CV_AA);
				cv::rectangle(colorImgCV, searchROITlCorner, searchROIBrCorner, CV_RGB(0, 255, 0), 1, CV_AA);
				
				std::vector<dlib::full_object_detection> shapes;
				for (unsigned long i = 0; i < faces.size(); ++i)
					shapes.push_back(pose_model(imgQuarter_dlib, faces[i]));

				if (shapes.size() != 0)
				{
					for (int i = 0; i < shapes[0].num_parts(); i++)
					{
						cv::Point2f landmarkPt = cv::Point2f(ResizeScale * shapes[0].part(i).x(), ResizeScale * shapes[0].part(i).y());
						cv::circle(colorImgCV, landmarkPt, 0, CV_RGB(255, 0, 0), 3, CV_AA);
					}
				}
			}

			cv::imshow("Image", colorImgCV);
			cv::imshow("QuarterImage", grayImgCVQuarter);
			cv::waitKey(1);
		}
	}

	system("pause");
}