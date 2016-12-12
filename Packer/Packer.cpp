#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace cv;
using namespace std;

const int MAX_FILE_SIZE = 3000000;

void processFace(vector<Mat> &variants, CascadeClassifier &classifier, Mat &variant, int im_width, int im_height)
{
	int side_length = 300;
	int square_x = variant.cols / 2 - side_length / 2;
	int square_y = variant.rows / 2 - side_length / 2;
	vector<Rect> faces;
	cvtColor(variant, variant, CV_BGR2GRAY);
	classifier.detectMultiScale(variant, faces);
	for (int i = 0; i < (int)faces.size(); i++) {
		Rect face = faces[i];
		if (face.width < 120 || face.height < 120)
			continue;
		if (face.x < square_x || face.x + face.width > square_x + side_length || face.y < square_y || face.y + face.height > square_y + side_length)
			continue;
		Mat result_1 = variant(faces[0]);
		resize(result_1, result_1, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
		variants.push_back(result_1);
	}
}

void rotateImage(Mat &image, double angle)
{
	Point2f pt(image.cols / 2, image.rows / 2);
	Mat r = getRotationMatrix2D(pt, angle, 1.0);
	warpAffine(image, image, r, image.size());
}

vector<Mat> generateVariants(Mat original, int im_width, int im_height)
{
	CascadeClassifier classifier;
	classifier.load("haarcascade_frontalface_alt.xml");

	vector<Mat> variants;

	for (int i = 1; i <= 8; i++) {
		Mat variant = original.clone();
		double alpha = (double)rand() / (double)RAND_MAX * (2.5 - 1.0) + 1.0;
		double beta = (double)rand() / (double)RAND_MAX * 70.0;
		double angle = (double)rand() / (double)RAND_MAX * 30.0 - 15.0;
		std :: cout << alpha << " " << beta << " " << angle << endl;
		variant.convertTo(variant, -1, alpha, beta);
		rotateImage(variant, angle);
		processFace(variants, classifier, variant, im_width, im_height);
	}

	return variants;
}

int main(int argc, char** argv)
{
	srand((unsigned)time(NULL));

	VideoCapture capture(0);
	if (!capture.isOpened()) {
		cout << "Capture Device ID " << 0 << " cannot be opened!" << endl;
		return -1;
	}

	std :: cout << capture.get(CV_CAP_PROP_FPS) << endl;

	CascadeClassifier classifier;
	classifier.load("haarcascade_frontalface_alt.xml");

	vector<Mat> images;
	vector<Mat> testSample;
	vector<int>labels;

	for (int i = 1; i <= 20; i++) {
		for (int j = 1; j <= 10; j++) {
			Mat ima = imread(format("Faces\\s%d\\%d.pgm", i, j), 0);
			if (!ima.data) {
				std :: cout << "Failed to read file data!" << endl;
				return 0;
			}
			images.push_back(ima);
			labels.push_back(i);
		}
	}

	int im_width = images[0].cols;
	int im_height = images[0].rows;

	Mat frame;
	Mat reserve;

	int counter = 0;

	while (true) {
		capture >> frame;
		Mat original = frame.clone();
		frame.copyTo(reserve);

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

		Mat originalCopy = original.clone();
		string box_text;
		box_text = "Align your face with the blue box.";
		putText(originalCopy, box_text, Point(20, 20), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 255), 1.5);
		imshow("Face Recognizer", originalCopy);
		char key = (char)waitKey(10);
		if (key == 27) {
			break;
		}

		Mat gray;
		cvtColor(original, gray, CV_BGR2GRAY);
		vector<Rect_<int>> faces;
		classifier.detectMultiScale(gray, faces);
		
		bool flag = false;

		for (int i = 0; i < faces.size(); i++) {
			Rect face_i = faces[i];
			if (face_i.x < square_x || face_i.x + face_i.width > square_x + side_length || face_i.y < square_y || face_i.y + face_i.height > square_y + side_length) {
				continue;
			}
			if (face_i.width < 120 || face_i.height < 120) {
				continue;
			}
		
			flag = true;

			Mat face = gray(face_i);
			Mat face_resized;
			rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
			Mat submat = original(face_i);
			blur(submat, submat, Size(10, 10), Point(-1, -1));
			// equalizeHist(face_resized, face_resized);

			resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
			images.push_back(face_resized);
			labels.push_back(21);
			counter++;
			imwrite(format("Faces\\s21\\collected_%d.pgm", counter), face_resized);
		}

		
		if (flag == true) {
			box_text = "One face was captured.";
		} else {
			box_text = "Align your face with the blue box.";
		}
		putText(original, box_text, Point(20, 20), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255, 0, 255), 1.5);

		imshow("Face Recognizer", original);
		key = (char)waitKey(10);
		if (key == 27) {
			break;
		}

		if (flag == true) {
			vector<Mat> variants = generateVariants(frame, im_width, im_height);
			if (counter < 20) {
				for (int k = 0; k < variants.size(); k++) {
					images.push_back(variants[k]);
					labels.push_back(21);
					counter++;
					imwrite(format("Faces\\s21\\collected_%d.pgm", counter), variants[k]);		
				}
			} else {
				for (int k = 0; k < variants.size(); k++) {
					testSample.push_back(variants[k]);
					counter++;
					imwrite(format("Faces\\s21\\collected_%d.pgm", counter), variants[k]);	
				}
			}
		}

		if (counter > 30) {
			std :: cout << "Your faces are collected succesfully! Password is being set, please wait..." << endl;
			break;
		}
	}

	Mat lastFrame;
	reserve.copyTo(lastFrame);
	lastFrame.convertTo(lastFrame, -1, 0.35, 0.0);
	string text = "Password is being set. Wait...";
	putText(lastFrame, text, Point(80, 230), FONT_HERSHEY_PLAIN, 1.8, CV_RGB(255, 0, 255), 1.5);
	imshow("Face Recognizer", lastFrame);
	waitKey(2000);

	Ptr<FaceRecognizer> LBPHModel = createLBPHFaceRecognizer();
	LBPHModel->train(images, labels);
	LBPHModel->save("LBPHXml.xml");

	double aveConfi = 0;
	int sampleCounter = 0;
	for (int i = 0; i < testSample.size(); i++) {
		Mat sample = testSample[i];
		double confi;
		int label;
		LBPHModel->predict(sample, label, confi);
		std :: cout << label << " " << confi << endl;
		if (label == 21) {
			aveConfi += confi;
			sampleCounter ++;
		}
	}
	
	ofstream fout("threshold.txt");
	fout << aveConfi / sampleCounter << endl;
	fout.close();

	std :: cout << "Face password is set successfully!" << endl;

	capture.release();
	cv :: destroyWindow("Face Recognizer");

	return 0;
}
