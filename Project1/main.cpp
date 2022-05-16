#include <iostream>
#include <vector>
#include <cmath>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image_abstract.h>
#include <opencv2/opencv.hpp>
#include <opencv2/gapi/s11n.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace dlib;
using namespace cv;

struct correspondens {
	std::vector<int> index;
};

//Нахождение ключевых точек на лице с помощью dlib

void faceLandmarkDetection(dlib::array2d<unsigned char>& img, shape_predictor sp, std::vector<Point2f>& landmark)
{
	dlib::frontal_face_detector detector = get_frontal_face_detector();
	std::vector<dlib::rectangle> dets = detector(img);

	full_object_detection shape = sp(img, dets[0]);

	for (unsigned int i = 0; i < shape.num_parts(); ++i)
	{
		float x = shape.part(i).x();
		float y = shape.part(i).y();
		landmark.push_back(Point2f(x, y));
	}

}

//Нахождение точек выпуклой оболочки с помощью триангуляции Делоне


void delaunayTriangulation(const std::vector<Point2f>& hull, std::vector<correspondens>& delaunayTri, Rect rect)
{

	cv::Subdiv2D subdiv(rect);
	for (int it = 0; it < hull.size(); it++)
		subdiv.insert(hull[it]);
	std::vector<Vec6f> triangleList;
	subdiv.getTriangleList(triangleList);

	for (size_t i = 0; i < triangleList.size(); ++i)
	{

		std::vector<Point2f> pt;
		correspondens ind;
		Vec6f t = triangleList[i];
		pt.push_back(Point2f(t[0], t[1]));
		pt.push_back(Point2f(t[2], t[3]));
		pt.push_back(Point2f(t[4], t[5]));

		if (rect.contains(pt[0]) && rect.contains(pt[1]) && rect.contains(pt[2]))
		{
			int count = 0;
			for (int j = 0; j < 3; ++j)
				for (size_t k = 0; k < hull.size(); k++)
					if (abs(pt[j].x - hull[k].x) < 1.0 && abs(pt[j].y - hull[k].y) < 1.0)
					{
						ind.index.push_back(k);
						count++;
					}
			if (count == 3)
				delaunayTri.push_back(ind);
		}
	}


}




// Применение Афинного преобразования, полученное с помощью srcdir и destdir к src

void applyAffineTransform(Mat& warpImage, Mat& src, std::vector<Point2f>& srcTri, std::vector<Point2f>& dstTri)
{
	// Нахождения Афинного преобразования с учётом пар треугольников
	Mat warpMat = getAffineTransform(srcTri, dstTri);

	// Применение Афинного преобразования к src
	warpAffine(src, warpImage, warpMat, warpImage.size(), cv::INTER_LINEAR, BORDER_REFLECT_101);
}




//Преобразование треугольников из одного изображения в другое


void warpTriangle(Mat& img1, Mat& img2, std::vector<Point2f>& t1, std::vector<Point2f>& t2) 
{

	Rect r1 = boundingRect(t1);
	Rect r2 = boundingRect(t2);

	// Смещение точек по левому верхнему углу соответствующих прямоугольников
	std::vector<Point2f> t1Rect, t2Rect;
	std::vector<Point> t2RectInt;
	for (int i = 0; i < 3; i++)
	{

		t1Rect.push_back(Point2f(t1[i].x - r1.x, t1[i].y - r1.y));
		t2Rect.push_back(Point2f(t2[i].x - r2.x, t2[i].y - r2.y));
		t2RectInt.push_back(Point(t2[i].x - r2.x, t2[i].y - r2.y)); // для fillConvexPoly

	}

	// Получение маски с помощью заполнения треугольниками
	Mat mask = Mat::zeros(r2.height, r2.width, CV_32FC3);
	fillConvexPoly(mask, t2RectInt, Scalar(1.0, 1.0, 1.0), 16, 0);

	// Применение warpImage к небольшим прямоугольным участкам
	Mat img1Rect;
	img1(r1).copyTo(img1Rect);

	Mat img2Rect = Mat::zeros(r2.height, r2.width, img1Rect.type());

	applyAffineTransform(img2Rect, img1Rect, t1Rect, t2Rect);

	multiply(img2Rect, mask, img2Rect);
	multiply(img2(r2), Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
	img2(r2) = img2(r2) + img2Rect;

}

//вычисление расстояния между точками
double GetDistance(int x1, int y1, int x2, int y2)
{
	return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

//вычисление пропорций лица
std::vector <double> GetFaceProportions(std::vector<cv::Point2f> points, std::vector<pair<int, int>> Ind) 
{
	std::vector < double > res, dist;
	double dist_main = GetDistance(points[Ind[0].first].x, points[Ind[0].first].y,  points[Ind[0].second].x, points[Ind[0].second].y);
	dist.push_back(GetDistance(points[Ind[1].first].x, points[Ind[1].first].y, points[Ind[1].second].x, points[Ind[1].second].y));
	dist.push_back(GetDistance(points[Ind[2].first].x, points[Ind[2].first].y, points[Ind[2].second].x, points[Ind[2].second].y));
	dist.push_back(GetDistance(points[Ind[3].first].x, points[Ind[3].first].y, points[Ind[3].second].x, points[Ind[3].second].y));
	dist.push_back(GetDistance(points[Ind[4].first].x, points[Ind[4].first].y, points[Ind[4].second].x, points[Ind[4].second].y));
	dist.push_back(GetDistance(points[Ind[5].first].x, points[Ind[5].first].y, points[Ind[5].second].x, points[Ind[5].second].y));
	dist.push_back(GetDistance(points[Ind[6].first].x, points[Ind[6].first].y, points[Ind[6].second].x, points[Ind[6].second].y));
	dist.push_back(GetDistance(points[Ind[7].first].x, points[Ind[7].first].y, points[Ind[7].second].x, points[Ind[7].second].y));
	dist.push_back(GetDistance(points[Ind[8].first].x, points[Ind[8].first].y,  points[Ind[8].second].x, points[Ind[8].second].y));
	dist.push_back(GetDistance(points[Ind[9].first].x, points[Ind[9].first].y, points[Ind[9].second].x, points[Ind[9].second].y));
	for (const auto& r : dist) {
		res.push_back(r / dist_main);
	}

	return res;
}

int main() 
{
	setlocale(0, "");
	system("cls");
	// Загрузка данных в программу
	dlib::array2d<unsigned char> imgDlib1, imgDlib2, imgDlib3;
	const char* file_1_path = "1.jpg"; // Путь к изображению из которого берётся лицо  
	const char* file_2_path = "2.jpg"; // Путь к изображению куда лицо помещается
	dlib::load_image(imgDlib1, file_1_path);
	dlib::load_image(imgDlib2, file_2_path);

	Mat imgCV1 = imread(file_1_path); 
	Mat imgCV2 = imread(file_2_path); 
	Mat imgCV1_copy = imgCV1.clone();
	Mat imgCV2_copy = imgCV2.clone();
	
	// Обнаружение ключевых точек лица
	shape_predictor sp;
	deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
	std::vector<Point2f> points1, points2, points3;

	faceLandmarkDetection(imgDlib1, sp, points1);
	faceLandmarkDetection(imgDlib2, sp, points2);

	// Нахождение convexHull
	Mat imgCV1Warped = imgCV2.clone();
	imgCV1.convertTo(imgCV1, CV_32F);
	imgCV1Warped.convertTo(imgCV1Warped, CV_32F);

	std::vector<Point2f> hull1;
	std::vector<Point2f> hull2;
	std::vector<int> hullIndex;

	cv::convexHull(points2, hullIndex, false, false);

	for (int i = 0; i < hullIndex.size(); i++)
	{
		hull1.push_back(points1[hullIndex[i]]);
		hull2.push_back(points2[hullIndex[i]]);
	}

	// Триангуляция Делоне	
	std::vector<correspondens> delaunayTri;
	Rect rect(0, 0, imgCV1Warped.cols, imgCV1Warped.rows);
	delaunayTriangulation(hull2, delaunayTri, rect);


	for (size_t i = 0; i < delaunayTri.size(); ++i)
	{
		std::vector<Point2f> t1, t2;
		correspondens corpd = delaunayTri[i];
		for (size_t j = 0; j < 3; ++j)
		{
			t1.push_back(hull1[corpd.index[j]]);
			t2.push_back(hull2[corpd.index[j]]);
		}

		warpTriangle(imgCV1, imgCV1Warped, t1, t2);
	}

	// вычисление маски
	std::vector<Point> hull8U;

	for (int i = 0; i < hull2.size(); ++i)
	{
		Point pt(hull2[i].x, hull2[i].y);
		hull8U.push_back(pt);
	}

	Mat mask = Mat::zeros(imgCV2.rows, imgCV2.cols, imgCV2.depth());
	fillConvexPoly(mask, &hull8U[0], hull8U.size(), Scalar(255, 255, 255));

	Rect r = boundingRect(hull2);
	Point center = (r.tl() + r.br()) / 2;

	Mat output;
	imgCV1Warped.convertTo(imgCV1Warped, CV_8UC3);
	seamlessClone(imgCV1Warped, imgCV2, mask, center, output, NORMAL_CLONE);

	// Запись результата в файл
	string filename = file_1_path;
	filename = filename + file_2_path;
	filename = filename + ".jpg";
	imwrite(filename, output);


	dlib::load_image(imgDlib3, filename);
	faceLandmarkDetection(imgDlib3, sp, points3);
	std::vector<pair<int, int>> Ind = { {0, 16}, {17, 39}, {26, 42}, {27, 30}, {31, 35}, {48, 54}, {17, 31}, {26, 35}, {48, 31}, {35, 54} }; // пары индексов точек в points для анализа

	std::vector<double> prop1 = GetFaceProportions(points1, Ind);
	std::vector<double> prop3 = GetFaceProportions(points3, Ind);
	
	double sm = 0.0, mx = 0.0, otn = 0.0;
	int pos = 0;
	for (size_t i = 0; i < prop1.size(); i++) 
	{	
		if (prop3[i] == 0) 
		{
			otn = 0.0;
		}
		else 
		{
			otn = abs(prop1[i] / prop3[i] - 1);
		}
		sm += otn;
		if (mx < otn) {
			mx = otn;
			pos = i;
		}
	}

	Mat output_copy = output.clone();
	Mat imgCV1_double_copy = imgCV1_copy.clone();

	cv::line(output, Point(points3[Ind[0].first].x, points3[Ind[0].first].y), Point(points3[Ind[0].second].x, points3[Ind[0].second].y), Scalar(255, 0, 0), 2);
	cv::line(imgCV1_copy, Point(points1[Ind[0].first].x, points1[Ind[0].first].y), Point(points1[Ind[0].second].x, points1[Ind[0].second].y), Scalar(255, 0, 0), 2);
	cv::line(output, Point(points3[Ind[pos].first].x, points3[Ind[pos].first].y), Point(points3[Ind[pos].second].x, points3[Ind[pos].second].y), Scalar(0, 255, 0), 2);
	cv::line(imgCV1_copy, Point(points1[Ind[pos].first].x, points1[Ind[pos].first].y), Point(points1[Ind[pos].second].x, points1[Ind[pos].second].y), Scalar(0, 255, 0), 2);

	for (size_t i = 1; i < 10; i++) {
		if (i == pos) { continue; }
		cv::line(output, Point(points3[Ind[i].first].x, points3[Ind[i].first].y), Point(points3[Ind[i].second].x, points3[Ind[i].second].y), Scalar(0, 0, 255), 2);
		cv::line(imgCV1_copy, Point(points1[Ind[i].first].x, points1[Ind[i].first].y), Point(points1[Ind[i].second].x, points1[Ind[i].second].y), Scalar(0, 0, 255), 2);
	}

	
	
	int effectiveness_1 = (1 - (sm / prop1.size())) * 100;
	int effectiveness_2 = (1 - mx) * 100;
	// Вывод на экран 5 изображений
	imshow("Face 1", imgCV1_double_copy);
	imshow("Face 2", imgCV2_copy);
	imshow("Final face", output_copy);
	imshow("Face 1 with lines", imgCV1_copy);
	imshow("Flinal Face with lines", output);
	system("cls");
	std::cout << "-------------------------------\n";
	std::cout << "абсолютная эффективность - " << effectiveness_2 << "%\n";
	std::cout << "средняя эффективность - " << effectiveness_1 << "%\n";
	std::cout << "-------------------------------\n";
	waitKey(0);
	destroyAllWindows();
	return 0;
}