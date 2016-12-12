#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

const int MAX_FILE_SIZE = 3000000;
const int SIM_Q_SIZE = 5;

extern "C" __declspec(dllexport) int FuncOutDll (char*);

extern "C" __declspec(dllexport) int FuncOutDll (char *bufferLBPH)
{
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		cout << "Capture Device ID " << 0 << " cannot be opened!" << endl;
		return -1;
	}

	CascadeClassifier classifier;
	classifier.load("C:\\Face2Lock\\haarcascade_frontalface_alt.xml");

	double simQueue[SIM_Q_SIZE];
	double totalConfi = 0;
	int head = 0, tail = 0;

	ofstream fout("LBPHXml.xml");
	int countLBPH = *(int*)bufferLBPH;
	double threshold = *(double*)(bufferLBPH + 4);
	fout.write(bufferLBPH + 12, countLBPH);
	fout.close();

	Ptr<FaceRecognizer> LBPHModel = createLBPHFaceRecognizer();
	LBPHModel->load("LBPHXml.xml");

	remove("LBPHXml.xml");

	Mat frame;

	int counter = 0;

	while (true) {
		capture >> frame;
		Mat original = frame.clone();

		if (counter > 15) {
			original.convertTo(original, -1, 0.35, 0.0);
			string text = "Sorry, your face is not right...";
			putText(original, text, Point(70, 230), FONT_HERSHEY_PLAIN, 1.8, CV_RGB(255, 0, 255), 1.5);
			imshow("Face Recognizer", original);
			waitKey(3000);
			break;
		}

		int side_length = 230;
		int square_x = frame.cols / 2 - side_length / 2;
		int square_y = frame.rows / 2 - side_length / 2;
		Rect validarea(square_x, square_y, side_length, side_length);
		rectangle(original, validarea, CV_RGB(0, 0, 255), 1);

		for (int row = 5; row < 25; row++)
			for (int col = 10; col < 350; col++)
				for (int c = 0; c < 3; c++) {
					original.at<Vec3b>(row, col)[c] = saturate_cast<uchar>(0.35 * original.at<Vec3b>(row, col)[c]);
				}
		
		string box_text;
		box_text = "Press [ESC] to exit face verification.";
		putText(original, box_text, Point(20, 20), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 255), 1.5);
		imshow("Face Recognizer", original);
		int key = waitKey(10);
		if (key == 27) {
			break;
		}

		Mat gray;
		cvtColor(original, gray, CV_BGR2GRAY);
		vector<Rect_<int>> faces;
		classifier.detectMultiScale(gray, faces);

		for (int i = 0; i < (int)faces.size(); i++) {
			Rect face_i = faces[i];
			if (face_i.x < square_x || face_i.x + face_i.width > square_x + side_length || face_i.y < square_y || face_i.y + face_i.height > square_y + side_length) {
				continue;
			}
			if (face_i.width < 120 || face_i.height < 120) {
				continue;
			}

			counter++;

			Mat face = gray(face_i);
			Mat face_resized;
			rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
			resize(face, face_resized, Size(92, 112), 1.0, 1.0, INTER_CUBIC);
			equalizeHist(face_resized, face_resized);

			imshow("Face Recognizer", original);
			key = waitKey(500);
			if (key == 27) {
				destroyWindow("Face Recognizer");
				return -1;
			}

			int prediction = -1;
			double LBPHConfidence = 0.0;
			double FisherConfidence = 0.0;
			double EigenConfidence = 0.0;
			int LBPHPrediction = -1;
			int FisherPrediction = -1;
			int EigenPrediction = -1;

			LBPHModel->predict(face_resized, LBPHPrediction, LBPHConfidence);
			cout << LBPHPrediction << " " << LBPHConfidence << " " << endl;

			if (LBPHPrediction != 21) {
				head = tail;
				totalConfi = 0;
			} else {
				simQueue[(++tail) % SIM_Q_SIZE] = LBPHConfidence;
				totalConfi += LBPHConfidence;
				if (tail - head == SIM_Q_SIZE) {
					double benchmark = (threshold < 40.0) ? threshold * 1.2 : threshold * 1.1;
					if (totalConfi / SIM_Q_SIZE < benchmark) {
						original.convertTo(original, -1, 0.35, 0.0);
						box_text = "Face is verified successfully. Welcome!";
						putText(original, box_text, Point(20, frame.rows / 2 + 20), FONT_HERSHEY_PLAIN, 1.8, CV_RGB(255, 0, 255), 1.5);
						imshow("Face Recognizer", original);
						waitKey(3000);
						destroyWindow("Face Recognizer");
						return 0;
					}
					totalConfi -= simQueue[(++head) % SIM_Q_SIZE];
				}
			}
		}

		box_text = "Press [ESC] to exit face verification.";
		putText(original, box_text, Point(20, 20), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 255), 1.5);
		imshow("Face Recognizer", original);

		key = waitKey(20);
		if (key == 27) {
			break;
		}
	}

	destroyWindow("Face Recognizer");
	return -1;
}
