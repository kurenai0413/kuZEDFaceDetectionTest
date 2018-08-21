#include <sl/Camera.hpp>
#include <chrono>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <opencv2/opencv.hpp>

#define ResizeScale		  2
#define SearchRegionScale 0.2

void main()
{
	cv::VideoCapture camCap(0);

	int capWidth  = camCap.get(cv::CAP_PROP_FRAME_WIDTH);
	int capHeight = camCap.get(cv::CAP_PROP_FRAME_HEIGHT);

	std::cout << "Resolution: (" << capWidth << " x " << capHeight << ")" << std::endl;

	cv::Mat colorImgCV;
	cv::Mat colorImgCVLeft;
	cv::Mat colorImgCVQuarter;
	cv::Mat grayImgCVQuarter;

	std::vector<cv::Rect>	faces;
	std::chrono::high_resolution_clock::time_point timeStamp[2];

	timeStamp[0] = std::chrono::high_resolution_clock::now();

	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor pose_model;
	dlib::deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

	cv::Rect leftImgRect(0, 0, 0.5 * capWidth, capHeight);

	if (camCap.isOpened())
	{
		std::cout << "Camera opened." << std::endl;
		
		while (true)
		{
			timeStamp[1] = std::chrono::high_resolution_clock::now();
			auto interval = std::chrono::duration_cast<std::chrono::milliseconds>(timeStamp[1] - timeStamp[0]);
			timeStamp[0] = timeStamp[1];
			std::cout << "Frame time: " << interval.count() << "ms, FPS: " << (float)1000 / (float)interval.count() << std::endl;

			camCap >> colorImgCV;
			colorImgCVLeft = colorImgCV(leftImgRect);

			float scale = (float)1 / (float)ResizeScale;

			cv::resize(colorImgCVLeft, colorImgCVQuarter, cv::Size(scale * colorImgCVLeft.cols, scale * colorImgCVLeft.rows));
			cv::Mat colorImgCVQuarterBGR;
			cv::cvtColor(colorImgCVQuarter, grayImgCVQuarter, CV_BGRA2GRAY);
			dlib::cv_image<uchar> imgQuarter_dlib(grayImgCVQuarter);

			// Detect faces 
			std::vector<dlib::rectangle> faces = detector(imgQuarter_dlib);

			if (faces.size() != 0)
			{
				cv::Point2f tlCorner = cv::Point2f(ResizeScale * (float)faces[0].left(), ResizeScale * (float)faces[0].top());
				cv::Point2f brCorner = cv::Point2f(ResizeScale * (float)faces[0].right(), ResizeScale * (float)faces[0].bottom());

				int originalFaceWidth = (int)abs(faces[0].left() - faces[0].right());
				int originalFaceHeight = (int)abs(faces[0].top() - faces[0].bottom());

				dlib::rectangle searchROI((long)(faces[0].left() - SearchRegionScale * originalFaceWidth),
					(long)(faces[0].top() - SearchRegionScale * originalFaceHeight),
					(long)(faces[0].right() + SearchRegionScale * originalFaceWidth),
					(long)(faces[0].bottom() + SearchRegionScale * originalFaceHeight));

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

			cv::imshow("Image", colorImgCVLeft);
			cv::imshow("QuarterImage", grayImgCVQuarter);
			cv::waitKey(1);
		}
	}

	
	system("pause");
}