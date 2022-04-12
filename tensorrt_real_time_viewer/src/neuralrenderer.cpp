#include "../include/neuralrenderer.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <sstream>

#include <GL/gl.h>
#include "GL/platform/Application.h"

#include "../include/basicbuffermanager.h"
#include "../include/camera.h"
#include "../include/cuda/donerf_cuda_kernels.cuh"
#include "../include/imagegenerator.h"
#include "../include/interoprenderbuffer.h"
#include "../include/helper.h"
#include "../include/settings.h"


#pragma pack(push, 2)
struct BmpHeader {
	char bitmapSignatureBytes[2] = { 'B', 'M' };
	uint32_t sizeOfBitmapFile = 54 + 400*400*3;
	uint32_t reservedBytes = 0;
	uint32_t pixelDataOffset = 54;
};

struct BmpInfoHeader {
	uint32_t sizeOfThisHeader = 40;
	int32_t width = 400; // in pixels
	int32_t height = 400; // in pixels
	uint16_t numberOfColorPlanes = 1; // must be 1
	uint16_t colorDepth = 24;
	uint32_t compressionMethod = 0;
	uint32_t rawBitmapDataSize = 0; // generally ignored
	int32_t horizontalResolution = 3780; // in pixel per meter
	int32_t verticalResolution = 3780; // in pixel per meter
	uint32_t colorTableEntries = 0;
	uint32_t importantColors = 0;
};
#pragma pack(pop)

NeuralRenderer::NeuralRenderer(Settings& settings, Config& config, GL::platform::Window& window, Camera& camera, 
                               int opengl_version_major, int opengl_version_minor) 
  : GL::platform::Renderer()
  , settings(settings), config(config), window(window), camera(camera), img_gen(settings, config)
  , context(window.createContext(opengl_version_major, opengl_version_minor, true))
  , context_scope(context, window)
{ 
  window.attach(this);
}

NeuralRenderer::~NeuralRenderer() 
{
  if (this->buffer_manager != nullptr)
    delete this->buffer_manager;
}

bool NeuralRenderer::init()
{
  std::cout << "Model Path: " << settings.model_path << std::endl;

  int w = settings.width;
  int h = settings.height;

  float fov = config.fov;

  std::vector<int> num_in_features;
  std::vector<int> num_out_features;

  for (int x = 0; x < config.posEnc.size(); x++)
  {
    Encoding enc;
    encodings.push_back(enc);

    bool res = false;
    if(config.inFeatures[x] != "SpherePosDir")
      res = encodings[x].load(config.posEnc[x], int(config.posEncArgs[x][0]), int(config.posEncArgs[x][1]));
    else
      res = encodings[x].load(config.posEnc[x], int(config.posEncArgs[x][0]), int(config.posEncArgs[x][1]), int(config.raySampleInput[x]) );
    if (!res)
      return false;

    num_in_features.push_back(encodings[x].getNumFeatures());
  }

  int rgba_raymarch_pre = 0;
  for (int x = 0; x < config.outFeatures.size(); x++)
  {
    if(config.outFeatures[x] == "RGB")
      num_out_features.push_back(3);

    if (config.outFeatures[x] == "RGBARayMarch")
    {
      num_out_features.push_back(4 * (config.numRaymarchSamples[x] + rgba_raymarch_pre) );
      rgba_raymarch_pre = config.numRaymarchSamples[x];
    }

    if (config.outFeatures[x] == "ClassifiedDepth")
      num_out_features.push_back(config.raySampleInput[x]);
  }

  for (int x = 1; x < config.inFeatures.size(); x++)
  {
    if (config.inFeatures[x] == "RayMarchFromPoses")
    {
      feature_sets.push_back((FeatureSet*) new RayMarchFromPoses(settings, config));
      ((RayMarchFromPoses*)feature_sets[x-1])->create(w, h, settings.batch_size, &encodings[x], x);
    }
    if (config.inFeatures[x] == "RayMarchFromCoarse")
    {
      feature_sets.push_back((FeatureSet*) new RayMarchFromCoarse(settings, config));
      ((RayMarchFromCoarse*)feature_sets[x - 1])->create(w, h, settings.batch_size, &encodings[x], x);
    }
  }

  if (!camera.init(config.inFeatures[0], fov, &encodings[0]))
    return false;

  CUDA_CHECK;

  if (!img_gen.load(num_in_features, num_out_features))
    return false;

  CUDA_CHECK;

  if (settings.is_debug)
    buffer_manager = new BasicBufferManager();
  else
    buffer_manager = new InteropRenderbuffer();


  if (!buffer_manager->init(w, h))
    return false;

  CUDA_CHECK;

  return true;
}

void NeuralRenderer::resize(int width, int height)
{
  window_size = glm::ivec2(width, height);
}

void NeuralRenderer::render()
{
  auto current = std::chrono::steady_clock::now();
  float ms = (float)std::chrono::duration_cast<std::chrono::microseconds> (current - this->last).count();
  this->last = current;

  this->buffer_manager->map();

  // to counter frame-rate dependent timing (e.g. always 33.3 ms for 30 fps)
  cudaDeviceSynchronize();

  auto current_bi = std::chrono::steady_clock::now();
  this->img_gen.inference(this->camera, this->buffer_manager->getCurrentBufferSurface(),
                          this->settings.batch_size, this->config.numRaymarchSamples[this->config.numRaymarchSamples.size()-1], 
                          this->getFeatureSet(), this->getEncoding());
  auto current_ai = std::chrono::steady_clock::now();

  float ms2 = (float)std::chrono::duration_cast<std::chrono::microseconds> (current_ai - current_bi).count();

  this->buffer_manager->unmap();

  stringstream ss;
  ss << "NeuralRenderer iter: " << std::fixed << std::setprecision(2) << ms / 1000.0f 
     << " (inference: " << ms2 / 1000.0f << ") [ms]";


	window.title(ss.str().c_str());
  
  this->buffer_manager->swap(); 
  this->buffer_manager->blit(window_size.x, window_size.y);

	swapBuffers();

  CUDA_CHECK;

  if (settings.write_images)
    writeImageToFile();
}

void NeuralRenderer::writeImageToFile()
{
	std::stringstream output_file_path;
	output_file_path << settings.model_path << "out.bmp";
	std::ofstream fout(output_file_path.str(), std::ios::binary);

  int w = settings.width;
  int h = settings.height;

  std::unique_ptr<unsigned char[]> image {std::make_unique<unsigned char[]>(w * h * 4)};
  
  cudaMemcpy2DFromArray(image.get(), w * sizeof(uchar4), this->buffer_manager->getCurrentBuffer(), 
                        0, 0, w * sizeof(uchar4), h, cudaMemcpyDeviceToHost);
  CUDA_CHECK;                      

  BmpHeader bmpHeader;
  bmpHeader.sizeOfBitmapFile = bmpHeader.pixelDataOffset + h * w * 3;

  BmpInfoHeader bmpInfoHeader;
  bmpInfoHeader.height = h;
  bmpInfoHeader.width = w;

  fout.write((char*)&bmpHeader, 14);
  fout.write((char*)&bmpInfoHeader, 40);

  // writing pixel data
  for (int y = 0; y < w; y++)
  {
    for (int x = 0; x < h; x++)
    {
      int i = (w - y - 1) * h + x; // flip horizontally

      fout.write((char*)image.get() + (4 * i + 2), 1);
      fout.write((char*)image.get() + (4 * i + 1), 1);
      fout.write((char*)image.get() + (4 * i),     1);
    }
  }
  fout.close();
}

///////////////////////////////////////////////////////////////////////////////

void NeuralRenderer::swapBuffers()
{
  context_scope.swapBuffers();
}

///////////////////////////////////////////////////////////////////////////////

void NeuralRenderer::close()
{
	GL::platform::quit();
}

///////////////////////////////////////////////////////////////////////////

void NeuralRenderer::destroy()
{
	GL::platform::quit();
}

///////////////////////////////////////////////////////////////////////////

void NeuralRenderer::move([[maybe_unused]]int x, [[maybe_unused]]int y)
{
}