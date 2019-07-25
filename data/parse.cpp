#include <cstdlib>
#include <random>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <vector>

#define real float

bool primal = false;
bool sparse = true;
uint64_t samples;
uint64_t features;
std::string inputFile;
std::string outputFile;

void readDataSparse(std::string fileName,
    std::vector<std::vector<real>> &x, std::vector<real> &y,
    std::vector<std::vector<uint32_t>> &xidx,
    uint64_t samples, uint64_t features){
  std::string line;
  std::ifstream fileStream;
  fileStream.open(fileName);
  std::string subStr, dataStr, idxStr;
  uint64_t split, j, idx;
  real val;
  if(fileStream.is_open()){
    for(uint64_t i=0; i<samples; ++i){
      std::getline(fileStream, line);
      std::istringstream ss(line);
      ss >> subStr;
      y.push_back(stod(subStr));
      x.push_back(std::vector<real>());
      xidx.push_back(std::vector<uint32_t>());
      while(ss >> subStr){
        if(subStr.length() >= 3){
          split = subStr.find(':');
          x.at(i).push_back(stod(subStr.substr(split + 1,
              subStr.length())));
          xidx.at(i).push_back(stoi(subStr.substr(0, split)));
        }
      }
    }
    fileStream.close();
  } else {
    std::cerr << "File not found!\n";
    exit(0);
  }
}

void readDataPrimal(std::string fileName, real* x, real* y,
    uint64_t samples, uint64_t features){
  std::string line;
  std::ifstream fileStream;
  fileStream.open(fileName);
  std::string subStr, dataStr, idxStr;
  std::memset(x, 0, sizeof(real) * samples * features);
  uint64_t split, j, idx;
  if(fileStream.is_open()){
    for(uint64_t i=0; i<samples; ++i){
      std::getline(fileStream, line);
      std::istringstream ss(line);
      ss >> subStr;
      y[i] = stod(subStr);
      while(ss >> subStr){
        if(subStr.length() >= 3){
          split = subStr.find(':');
          idxStr = subStr.substr(0, split);
          dataStr = subStr.substr(split + 1, subStr.length());
          x[samples * (stoi(idxStr) - 1) + i] = stod(dataStr);
        }
      }
    }
    fileStream.close();
  }
}

void readDataDual(std::string fileName, real* x, real* y,
    uint32_t samples, uint32_t features){
  std::string line;
  std::ifstream fileStream;
  fileStream.open(fileName);
  std::memset(x, 0, sizeof(real) * samples * features);
  std::string subStr, dataStr, idxStr;
  uint64_t split, j, idx;
  if(fileStream.is_open()){
    for(uint64_t i=0; i<samples; ++i){
      uint64_t row = i * features;
      std::getline(fileStream, line);
      std::istringstream ss(line);
      ss >> subStr;
      y[i] = stod(subStr);
      while(ss >> subStr){
        if(subStr.length() >= 3){
          split = subStr.find(':');
          idxStr = subStr.substr(0, split);
          dataStr = subStr.substr(split + 1, subStr.length());
          x[row + (stoi(idxStr) - 1)] = stod(dataStr);
        }
      }
    }
    fileStream.close();
  }
}

void print_usage_and_exit() {
  std::cout << "USAGE\n"
  << "parse <input_file_name> <n_samples> <n_features> <output_file_name> <primal|dual> <sparse|dense>\n"
  exit(-1);
}

void parse_params(int argc, char *argv[]) {
  if (argc != 7)
    print_usage_and_exit();
  inputFile = argv[1];
  samples = std::atoi(argv[2]);
  features = std::atoi(argv[3]);
  if (samples <= 0 || features <= 0)
    print_usage_and_exit();
  outputFile = argv[4];
  if (!std::strcmp(argv[5], "primal"))
    primal = true;
  else if (!std::strcmp(argv[5], "dual"))
    primal = false;
  else
    print_usage_and_exit();
  if (!std::strcmp(argv[6], "sparse"))
    sparse = true;
  else if (!std::strcmp(argv[6], "dense"))
    sparse = false;
  else
    print_usage_and_exit();
}

int main(int argc, char *argv[]){
  parse_params(argc, argv);
  std::string dualSuffix = primal ? "" : "_dual";
	uint32_t nnz;
  if (sparse) {
    std::vector<std::vector<real>> x;
    std::vector<std::vector<uint32_t>> xidx;
    std::vector<real> y;
    readDataSparse(inputFile, x, y, xidx, samples, features);
    std::cout << "Finished reading." << std::endl;
    std::vector<std::vector<real>> x_t;
    std::vector<std::vector<uint32_t>> xidx_t;
    if (primal) {
      for(uint64_t i=0; i<features; ++i){
        x_t.push_back(std::vector<real>());
        xidx_t.push_back(std::vector<uint32_t>());
      }
      for(uint64_t i = 0; i < x.size(); ++i){
        for(uint64_t j = 0; j < x.at(i).size(); ++j){
          x_t.at(xidx.at(i).at(j) - 1).push_back(x.at(i).at(j));
          xidx_t.at(xidx.at(i).at(j) - 1).push_back(i + 1);
        }
      }
      std::cout << "Finished transpose." << std::endl;
    }
    
    std::ofstream xStream(outputFile + dualSuffix + "_sparseX",
        std::ios::out | std::ios::binary);
    if(!xStream){
      std::cout << "Could not open X." << std::endl;
      return 1;
    }
    if (primal) {
      for(uint64_t i = 0; i < x_t.size(); ++i){
        nnz = x_t.at(i).size();
        xStream.write(reinterpret_cast<const char*>(&nnz),
            std::streamsize(sizeof(uint32_t)));
        if(nnz > 0){
          for (uint64_t j = 0; j < nnz; ++j)
            xidx_t.at(i).at(j) -= 1;
          xStream.write(
              reinterpret_cast<const char*>(xidx_t.at(i).data()),
              std::streamsize(nnz * sizeof(uint32_t)));
          xStream.write(
              reinterpret_cast<const char*>(x_t.at(i).data()),
              std::streamsize(nnz * sizeof(real)));
        }
      }
    }
    else {
      for(uint64_t i = 0; i < x.size(); ++i){
        nnz = x.at(i).size();
        xStream.write(reinterpret_cast<const char*>(&nnz),
            std::streamsize(sizeof(uint32_t)));
        if(nnz > 0){
          for (uint64_t j = 0; j < nnz; ++j)
            xidx.at(i).at(j) -= 1;
          xStream.write(
              reinterpret_cast<const char*>(xidx.at(i).data()),
              std::streamsize(nnz * sizeof(uint32_t)));
          xStream.write(
              reinterpret_cast<const char*>(x.at(i).data()),
              std::streamsize(nnz * sizeof(real)));
        }
      }
    }
    xStream.close();
    std::ofstream yStream(outputFile + "Y",
        std::ios::out | std::ios::binary);
    if(!yStream){
        std::cout << "Could not open Y." << std::endl;
        return 0;
    }
    yStream.write(reinterpret_cast<const char*>(y.data()),
        std::streamsize(samples * sizeof(real)));
    yStream.close();
    std::cout << "Finished writing.\n";
  } else {
    real* x = new real[samples * features];
    real* y = new real[samples];
    if (primal)
      readDataPrimal(inputFile, x, y, samples, features);
    else
      readDataDual(inputFile, x, y, samples, features);
    std::cout << "Finished reading.\n" << std::endl;
    std::ofstream xStream(outputFile + dualSuffix + "X",
        std::ios::out | std::ios::binary);
    if(!xStream){
        std::cout << "Could not open X." << std::endl;
        return 1;
    }
    xStream.write(reinterpret_cast<const char*>(x),
        std::streamsize(samples * features * sizeof(real)));
    xStream.close();
    std::ofstream yStream(outputFile + "Y",
        std::ios::out | std::ios::binary);
    if(!yStream){
        std::cout << "Could not open Y." << std::endl;
        return 0;
    }
    yStream.write(reinterpret_cast<const char*>(y),
        std::streamsize(samples * sizeof(real)));
    yStream.close();
    std::cout << "Finished writing.\n";
    delete[] x;
    delete[] y;
  }
	return 0;
}
