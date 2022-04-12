#include "../include/config.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <locale>
#include <functional>

Config::Config()
{
}
Config::~Config()
{
}

bool readIntArray(std::vector<int> & vec, std::string del , std::string value)
{
  size_t start = value.find("[");
  size_t end = value.find("]");

  if (start == std::string::npos)
  {
    std::cout << "start of array not found"<<std::endl;
    return false;
  }
  if (end == std::string::npos)
  {
    std::cout << "end of array not found" << std::endl;
    return false;
  }
  if (start + 1 == end)
  {
    std::cout << "array empty" << std::endl;
    return true;
  }

  std::string s = value.substr(start+1, end-1);
  std::string delimiter = del;

  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delimiter)) != std::string::npos) 
  {
    token = s.substr(0, pos);
    vec.push_back(std::stoi(token));
    s.erase(0, pos + delimiter.length());
  }
  vec.push_back(std::stoi(s));

  return true;
}

bool readFloatArrayClean(std::vector<float>& vec, std::string del, std::string value)
{ 
  std::string s = value;
  std::string delimiter = del;

  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delimiter)) != std::string::npos)
  {
    token = s.substr(0, pos);
    vec.push_back(std::stof(token));
    s.erase(0, pos + delimiter.length());
  }
  vec.push_back(std::stof(s));

  return true;
}


bool readFloatArray(std::vector<float>& vec, std::string del, std::string value)
{
  size_t start = value.find("[");
  size_t end = value.find("]");

  if (start == std::string::npos)
  {
    std::cout << "start of array not found" << std::endl;
    return false;
  }
  if (end == std::string::npos)
  {
    std::cout << "end of array not found" << std::endl;
    return false;
  }
  if (start + 1 == end)
  {
    std::cout << "array empty" << std::endl;
    return true;
  }

  std::string s = value.substr(start + 1, end - 1);
  std::string delimiter = del;

  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delimiter)) != std::string::npos)
  {
    token = s.substr(0, pos);
    vec.push_back(std::stof(token));
    s.erase(0, pos + delimiter.length());
  }
  vec.push_back(std::stof(s));

  return true;
}

bool readMultiFloatArray(std::vector <std::vector<float>>& vec, std::string del, std::string value)
{
  size_t start = value.find("[");
  size_t end = value.find("]");

  if (start == std::string::npos)
  {
    std::cout << "start of array not found" << std::endl;
    return false;
  }
  if (end == std::string::npos)
  {
    std::cout << "end of array not found" << std::endl;
    return false;
  }
  if (start + 1 == end)
  {
    std::cout << "array empty" << std::endl;
    return true;
  }

  std::string s = value.substr(start + 1, end - 1);

  std::string delimiter = del;

  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delimiter)) != std::string::npos)
  {
    token = s.substr(0, pos);    
    std::vector<float> tf;

    if (token == "none")
    {
      tf.push_back(4.0f);
      tf.push_back(0.0f);
    }
    else
      readFloatArrayClean(tf, "-", token);

    vec.push_back(tf);
    s.erase(0, pos + delimiter.length());
  }
  std::vector<float> tf;
  vec.push_back(tf);
  readFloatArrayClean(vec[vec.size() - 1], "-", s);

  return true;
}


bool readStringArray(std::vector<std::string>& vec, std::string del, std::string value)
{
  size_t start = value.find("[");
  size_t end = value.find("]");

  if (start == std::string::npos)
  {
    std::cout << "start of array not found" << std::endl;
    return false;
  }
  if (end == std::string::npos)
  {
    std::cout << "end of array not found" << std::endl;
    return false;
  }
  if (start + 1 == end)
  {
    std::cout << "array empty" << std::endl;
    return true;
  }

  std::string s = value.substr(start + 1, end - 1);
  std::string delimiter = del;

  size_t pos = 0;
  std::string token;
 
  while ((pos = s.find(delimiter)) != std::string::npos)
  {
    token = s.substr(0, pos);
    vec.push_back(token);
    s.erase(0, pos + delimiter.length());
  }
  vec.push_back(s);

  return true;
}


void Config::store(std::string key, std::string value)
{
  // strip whitespaces 
  key.erase(std::remove_if(key.begin(), key.end(), std::bind(std::isspace<char>, std::placeholders::_1, std::locale::classic())), key.end());
  value.erase(std::remove_if(value.begin(), value.end(), std::bind(std::isspace<char>, std::placeholders::_1, std::locale::classic())), value.end());

  if (key == "posEncArgs")
    readMultiFloatArray(posEncArgs, ",", value);

  if (key == "posEnc")
    readStringArray(posEnc, ",", value);
  if (key == "inFeatures")
    readStringArray(inFeatures, ",", value);
  if (key == "outFeatures")
    readStringArray(outFeatures, ",", value);

  if (key == "rayMarchSampler")
    readStringArray(rayMarchSampler, ",", value);
  if (key == "rayMarchNormalization")
    readStringArray(rayMarchNormalization, ",", value);
  if (key == "activation")
    readStringArray(activation, ",", value);

  
  if (key == "numRaymarchSamples")
    readIntArray(numRaymarchSamples, ",", value);

  if (key == "rayMarchSamplingStep")
    readFloatArray(rayMarchSamplingStep, ",", value);
  if (key == "rayMarchSamplingNoise")
    readFloatArray(rayMarchSamplingNoise, ",", value);
  if (key == "zNear")
    readFloatArray(zNear, ",", value);
  if (key == "zFar")
    readFloatArray(zFar, ",", value);
    
  if (key == "depth_range")
    readFloatArray(depthRange, ",", value);
  if (key == "view_cell_size")
    readFloatArray(viewcellSize, ",", value);
  if (key == "view_cell_center")
    readFloatArray(viewcellCenter, ",", value);

  if (key == "fov")
    fov = (float)atof(value.c_str());
  if (key == "max_depth")
    max_depth = (float)atof(value.c_str());

  if (key == "raySampleInput")
    readIntArray(raySampleInput, ",", value);

  if (key == "depthTransform")
    depthTransform = value;
}


bool Config::load(std::string file_name_)
{
  file_dir = file_name_;

  std::stringstream ss;
  ss << file_dir << "config.ini";
  file_name = ss.str();

  std::ifstream file(file_name);

  if (!file)
  {
    std::cout << "couldn't open config file " << file_name << std::endl;
    return false;
  }
  
  std::stringstream  is_file;
  is_file << file.rdbuf();

  std::cout << "reading config file" << file_name << std::endl;

  std::string line;
  while (std::getline(is_file, line))
  {
    std::istringstream is_line(line);
    std::string key;
    if (std::getline(is_line, key, '='))
    {
      std::string value;
      if (std::getline(is_line, value))
        store(key, value);
    }
  }
  std::cout << "read config file" << std::endl;

  if (!loadDatasetInfo(file_dir))
    return false;

  return true;
}


bool Config::loadDatasetInfo(std::string base_data_dir_)
{
  std::stringstream ss;
  ss << base_data_dir_ << "/dataset_info.txt";
 
  std::ifstream file(ss.str());

  if (!file)
  {
    std::cout << "couldn't open dataset info file " << ss.str() << std::endl;
    return false;
  }

  std::stringstream  is_file;
  is_file << file.rdbuf();

  std::cout << "reading dataset info file" << ss.str() << std::endl;

  std::string line;
  while (std::getline(is_file, line))
  {
    std::istringstream is_line(line);
    std::string key;
    if (std::getline(is_line, key, '='))
    {
      std::string value;
      if (std::getline(is_line, value))
        store(key, value);
    }
  }

  return true;
}