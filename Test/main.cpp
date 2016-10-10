#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE MCL_TEST
#include <boost/test/unit_test.hpp>
#include <boost/lexical_cast.hpp>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include <mcl/mcl.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iomanip>

using namespace std;

double inflat(double val)
{
	return std::pow(val, 1.5);
}

Eigen::MatrixXd readcsv(string file)
{
	std::ifstream csvfile(file);
	std::string line;
	size_t row = 0, col = 0;
	vector<double> vec;
	while (getline(csvfile, line))
	{
		std::istringstream s(line);
		std::string field;
		while (getline(s, field, ','))
			vec.push_back(boost::lexical_cast<double>(field));
		row++;
	}
	Eigen::MatrixXd mat(row, row);
	for (auto r = 0; r < row; r++)
		for (auto c = 0; c < row; c++)
			mat(r, c) = vec[r*row + c];
	
	return std::move(mat);
}

BOOST_AUTO_TEST_CASE(examples_csv)
{
	Eigen::MatrixXd mat(readcsv("example.csv"));

	std::map<size_t, vector<size_t>> results;
	auto result_cap = [&](size_t r, size_t c) { results[r].push_back(c); };

	auto start = std::chrono::system_clock::now();
	mcl_cpp::mcl mclob(mat, result_cap);
	auto m_res = mclob.cluster_mcl(3, 2, 60, 2);
	auto elap = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start).count();
	std::cout << "Elapsed=" << elap << " millisecs" << endl;

	for (auto s : results)
	{
		cout << "Cluster " << s.first << " [";
		for (auto c : s.second)
			cout << c << ",";
		cout << "]" << endl;
	}

	vector<size_t> expected_7 = { 5, 7, 20, 22, 27, 54, 57, 81, 125, 157, 181, 245, 251, 263, 284, 293 };
	BOOST_CHECK((results[7] == expected_7));
	
	vector<size_t> expected_11 = { 4, 9, 11, 14, 33, 34, 100, 113, 159, 172, 272 };
	BOOST_CHECK((results[11] == expected_11));

	vector<size_t> expected_28 = { 0, 2, 3, 8, 12, 13, 16, 17, 19, 24, 25, 28, 29, 30, 32, 35, 37, 40, 43, 44, 45, 46, 47, 51, 53, 56, 58, 59, 62, 63, 66, 68, 69, 70, 71, 72, 76, 79, 83, 87, 90, 91, 92, 94, 96, 97, 98, 99, 102, 104, 106, 107, 108, 109, 111, 112, 114, 115, 116, 117, 118, 119, 121, 124, 126, 131, 132, 133, 135, 136, 137, 138, 140, 141, 145, 147, 148, 149, 150, 152, 154, 156, 158, 160, 163, 164, 167, 169, 170, 171, 174, 176, 177, 178, 182, 184, 185, 186, 187, 190, 193, 195, 196, 202, 203, 205, 206, 207, 211, 212, 213, 216, 218, 219, 220, 221, 222, 225, 226, 227, 228, 229, 233, 237, 238, 239, 241, 247, 248, 250, 253, 254, 255, 257, 258, 259, 260, 261, 264, 265, 266, 269, 275, 277, 279, 280, 281, 282, 286, 289, 290, 291, 294, 296 };
	BOOST_CHECK((results[28] == expected_28));

	vector<size_t> expected_127 = { 1, 6, 21, 26, 31, 49, 52, 55, 61, 67, 75, 77, 82, 84, 85, 88, 93, 95, 123, 127, 128, 134, 139, 144, 151, 153, 173, 180, 188, 192, 198, 200, 208, 209, 215, 217, 240, 242, 246, 249, 276, 283, 295, 299 };
	BOOST_CHECK((results[127] == expected_127));

	vector<size_t> expected_223 = { 10, 18, 36, 38, 41, 48, 50, 64, 65, 73, 74, 78, 80, 86, 89, 101, 105, 110, 120, 122, 129, 130, 143, 146, 155, 161, 162, 165, 166, 168, 175, 179, 183, 189, 191, 197, 199, 201, 204, 214, 223, 224, 230, 232, 234, 235, 236, 243, 244, 252, 256, 262, 267, 268, 270, 271, 287, 288, 297, 298 };
	BOOST_CHECK((results[223] == expected_223));

	vector<size_t> expected_292 = { 15, 23, 39, 42, 60, 103, 142, 194, 210, 231, 273, 274, 278, 285, 292 };
	BOOST_CHECK((results[292] == expected_292));
}

BOOST_AUTO_TEST_CASE(my_test)
{
	Eigen::MatrixXd in(10.0*Eigen::MatrixXd::Random(5, 5).cwiseAbs());

	std::map<size_t, vector<size_t>> results;
	auto result_cap = [&](size_t r, size_t c) { results[r].push_back(c); };

	mcl_cpp::mcl mclob(in, result_cap);
	mclob.cluster_mcl(4, 1.5, 60, 2);

}
