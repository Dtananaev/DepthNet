 #include <pwd.h>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <curl/curl.h>
#include <dirent.h>
#include <cerrno>
#include <algorithm>
#include <fstream>
#include <sstream>

#include <png++/png.hpp>
#include <jpeglib.h>
#include <unistd.h>
#include <cmath>
using namespace std;
void GetLocalFileNames(const string &dir, vector<string> &file_list) {
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(dir.c_str())) == NULL) {
      cout << "Error(" << errno << ") opening " << dir << endl;
  }

  while ((dirp = readdir(dp)) != NULL) {
      file_list.push_back(dir + string(dirp->d_name));
  }
  closedir(dp);

  sort( file_list.begin(), file_list.end() );
  file_list.erase(file_list.begin()); //.
  file_list.erase(file_list.begin()); //..
}




const int kImageRows = 480;
const int kImageCols = 640;
const int kSampleFactor = 30;
const int kImageChannels = 3;
const int kFileNameLength = 24;
const int kTimeStampPos = 8;
const int kTimeStampLength = 12;

void GetDepthData( std::string file_name) {
  png::image< png::gray_pixel_16 > img(file_name.c_str(),png::require_color_space< png::gray_pixel_16 >());
//std::cout<<"read "<<file_name<<"\n";


  int index = 0;
  for (int i = 0; i < kImageRows; ++i) {
    for (int j = 0; j < kImageCols; ++j) {
      if(std::isnan(img.get_pixel(j, i))==true){
         std::cout<<"file_name["<<j<<"]["<<i<<"]=NaN"<<"\n"; 
         std::cin.get();
    }
//std::cout<<img.get_pixel(j, i)<<"\n";
    }
  }


}

int main(int argc, char **argv) {
  string local_dir;

  if (argc == 2) {
      local_dir=argv[1];
      std::vector<std::string> list;
      std::cout<<"Path to directory to unpack all dataset: "<<local_dir<<"\n";
      GetLocalFileNames(local_dir,list);
      for(int i=0;i<list.size();i++){
          GetDepthData( list[i]);
          //std::cin.get();
          std::cout<<i<<"\n";
    }  
    }


          return 0;

}
