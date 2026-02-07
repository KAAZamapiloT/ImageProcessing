#include<iostream>

#include<fstream>

#include <string>
#include<vector>

enum class ImageType{
  PNG,JPG,TIF,BMP
};


class ImageClass{
  public:

  ImageClass(std::string filepath):m_filepath(filepath){

  }

  private:
  ImageType type;
  std::string m_filepath;

  //std::vector<uint8_t> data;
  void load(){
      switch(type){
        case ImageType::PNG:
          loadPNG();
          break;
        case ImageType::JPG:
          loadJPG();
          break;
        case ImageType::TIF:
          loadTIF();
          break;
        case ImageType::BMP:
          loadBMP();
          break;
        default:
          throw std::runtime_error("Unsupported image type");
      }
  }


  void loadJPG(){

  }
  void loadPNG(){

  }
  void loadBMP(){

  }
  void loadTIF(){

  }

};
