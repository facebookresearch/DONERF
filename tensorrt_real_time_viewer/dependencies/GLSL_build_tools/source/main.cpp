//
//
//
//#include <iostream>
//#include <vector>
//#include <memory>
//#include <fstream>
//#include <stdexcept>
//#include <string>
//#include <cstring>
//
//#include "PTXObjectFile.h"
//#include "PTXBinary.h"
//
//#include "StdStreamLog.h"
//
//
//class invalid_argument_exception : public std::exception
//{
//public:
//  const char* what() const { return "invalid argument"; }
//};
//
//
//void help()
//{
//  std::cout << "ptxlink version 0.0" << std::endl;
//}
//
//int main(int argc, char* argv[])
//{
//  if (argc < 2)
//  {
//    std::cout << "no input given." << std::endl;
//    return 0;
//  }
//
//  std::vector<std::unique_ptr<PTX::ObjectFile>> objects;
//
//  const char* output_filename = nullptr;
//
//  try
//  {
//    StdStreamLog log;
//
//    PTX::Binary bin(log);
//
//    for (int i = 1; i < argc; ++i)
//    {
//      if (argv[i][0] == '-')
//      {
//        if (argv[i][1] != '\0')
//        {
//          if (strcmp(argv[i] + 1, "o") == 0)
//          {
//            ++i;
//            if (i >= argc)
//              throw std::runtime_error("-o missing output file");
//            output_filename = argv[i];
//          }
//          else if (strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "?") == 0)
//            help();
//          else
//            throw std::runtime_error(std::string("unknown argument: ") + argv[i]);
//        }
//        else
//          throw invalid_argument_exception();
//      }
//      else
//      {
//        objects.emplace_back(new PTX::ObjectFile(argv[i]));
//        bin.addObject(*objects.back());
//      }
//    }
//
//    if (output_filename == nullptr)
//      throw std::runtime_error("no output file specified");
//
//    bin.write(output_filename);
//  }
//  catch (std::exception& e)
//  {
//    std::cout << "error: " << e.what() << std::endl;
//    return -1;
//  }
//  catch (...)
//  {
//    std::cout << "unknown exception" << std::endl;
//    return -1;
//  }
//
//  return 0;
//}
