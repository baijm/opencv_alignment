#include <boost/filesystem.hpp>
//#include <boost/date_time/gregorian/gregorian.hpp>    
//#include <boost/date_time/posix_time/posix_time.hpp>

#include <iostream>
#include <string>

namespace fs = boost::filesystem;
using namespace std;

int main()
{
	fs::path new_path("/home/bjm/projects/opencv_alignment/test_fs");
	cout << "test boost::filesystem::create_directories" << endl;
	fs::create_directories(new_path);

	// test boost::filesystem
	fs::path test_path("/home/bjm/projects/opencv_alignment/config.yml");
	cout << "test boost::filesystem::exists" << endl;
	cout << "\treturn value = " << fs::exists(test_path) << endl;

	cout << "test boost::filesystem::is_regular_file" << endl;
	cout << "\treturn value = " << fs::is_regular_file(test_path) << endl;

	return 0;
}

/*
g++ -g src/test_boost.cpp -o test_boost -I /home/bjm/environments/boost_1_60_0/install/include -L /home/bjm/environments/boost_1_60_0/install/lib -lboost_system -lboost_filesystem
*/
