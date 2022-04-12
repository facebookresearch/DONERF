#include "../include/imagegenerator.h"
  
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include "../include/bmp.h"
#include "../include/config.h"
#include "../include/cuda/donerf_cuda_kernels.cuh"
#include "../include/encoding.h"
#include "../include/settings.h"

#include "EntropyCalibrator.h"


ImageGenerator::ImageGenerator(Settings& settings, Config& config)
  : settings(settings), config(config)
{
}

ImageGenerator::~ImageGenerator(){}

nvinfer1::ICudaEngine* loadEngine(const std::string& engine, int DLACore)
{
  std::ifstream engineFile(engine, std::ios::binary|std::ios::in);
  if (!engineFile)
  {
    std::cout << "Error opening engine file: " << engine << std::endl;
    return nullptr;
  }

  engineFile.seekg(0, engineFile.end);
  long int fsize = engineFile.tellg();
  engineFile.seekg(0, engineFile.beg);

  std::vector<char> engineData(fsize);
  engineFile.read(engineData.data(), fsize);
  if (!engineFile)
  {
    std::cout << "Error loading engine file: " << engine << std::endl;
    return nullptr;
  }

  IRuntime* runtime{ createInferRuntime(sample::gLogger.getTRTLogger()) };
  if (DLACore != -1)
  {
    runtime->setDLACore(DLACore);
  }

  return runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr) ;
}


bool saveEngine(std::shared_ptr<nvinfer1::ICudaEngine> engine, const std::string& fileName)
{
  std::ofstream engineFile(fileName, std::ios::binary|std::ios::out);
  if (!engineFile)
  {
    std::cout << "Cannot open engine file: " << fileName << std::endl;
    return false;
  }

  IHostMemory* serializedEngine{ engine->serialize() };
  if (serializedEngine == nullptr)
  {
    std::cout << "Engine serialization failed" << std::endl;
    return false;
  }

  engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
  return !engineFile.fail();
}

void setProfileDimensions2D(IOptimizationProfile* profile, const char* input_name, int min1, int max1, int value2)
{
  profile->setDimensions(input_name, OptProfileSelector::kMIN, Dims2(min1, value2));
  profile->setDimensions(input_name, OptProfileSelector::kOPT, Dims2(max1, value2));
  profile->setDimensions(input_name, OptProfileSelector::kMAX, Dims2(max1, value2));
}

bool ImageGenerator::initEngine(const std::string& model_path, 
                                const std::string& feature_in, const std::string& feature_out, const std::string& encoding, 
                                int num_in_features, int num_out_features, 
                                int count, int batch_size, int num_samples)
{
  int min_model_inputs = batch_size * num_samples;
  int max_model_inputs = batch_size * num_samples;

  std::stringstream ss_engine, ss_model;
  ss_model << "model" << count << ".onnx";
  ss_engine << model_path << "engine_" << count
            << "_min_" << min_model_inputs << "_max_" << max_model_inputs << ".engine";

  std::string model_name = ss_model.str();
  std::string engine_name = ss_engine.str();
  std::vector<std::string> dirs;
  dirs.push_back(model_path);


  auto builder = IGUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
  if (!builder)
  {
    return false;
  }

  const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = IGUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
  if (!network)
  {
    return false;
  }

  auto config = IGUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config)
  {
    return false;
  }

  if (feature_in == "SpherePosDir")
  {
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    setProfileDimensions2D(profile, "input_1", batch_size, batch_size, num_in_features);
    config->addOptimizationProfile(profile);
  }
  else if (feature_in == "RayMarchFromPoses")
  {
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    setProfileDimensions2D(profile, "input_1", min_model_inputs, max_model_inputs, num_in_features);
    config->addOptimizationProfile(profile);
  }
  else if (feature_in == "RayMarchFromCoarse")
  {
    IOptimizationProfile* profile = builder->createOptimizationProfile();
    setProfileDimensions2D(profile, "input_1", batch_size * (128 + 64), batch_size * (128 + 64), num_in_features);
    config->addOptimizationProfile(profile);
  }

  auto parser = IGUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
  if (!parser)
  {
    return false;
  }

  auto parsed = parser->parseFromFile(locateFile(model_name, dirs).c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
  if (!parsed)
  {
    return false;
  }

  printNetwork(network);

  if (builder->platformHasFastFp16())
    config->setFlag(BuilderFlag::kFP16);

  std::shared_ptr<nvinfer1::ICudaEngine> engine = 
    std::shared_ptr<nvinfer1::ICudaEngine>(loadEngine(engine_name, -1), samplesCommon::InferDeleter());
  if (!engine)
  {
    config->setMaxWorkspaceSize(8_GiB);
    config->setDefaultDeviceType(DeviceType::kGPU);

    engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), samplesCommon::InferDeleter());

    saveEngine(engine, engine_name);
  }

  if (!engine)
  {
    return false;
  }

  engines.push_back(engine);

  void* out;
  if (feature_in == "RayMarchFromCoarse")
    cudaMalloc(&out, batch_size * (128+64) * num_out_features * sizeof(float));
  else
    cudaMalloc(&out, batch_size * num_out_features * sizeof(float));

  d_outputs.push_back(out);

  std::vector<void*> bindings;
  if(count == 0)
    bindings.push_back(0);
  else
    bindings.push_back(d_outputs[count - 1]);

  bindings.push_back(d_outputs[count]);
  inference_bindings.push_back(bindings);

  contexts.push_back( IGUniquePtr<nvinfer1::IExecutionContext>(engines[engines.size()-1]->createExecutionContext()) );
  if (!contexts[contexts.size()-1])
  {
    return false;
  }    

  return true;
}


bool ImageGenerator::load(std::vector<int>num_in_features, std::vector<int> num_out_features)
{
  std::cout << "Loading num steps:" << config.inFeatures.size() << std::endl;

  for (int x = 0; x < config.inFeatures.size(); x++)
  {
    int samples = 0;

    if (config.numRaymarchSamples.size() > 0)
      samples = config.numRaymarchSamples[x];

    bool init_success = true;
    init_success = initEngine(settings.model_path, config.inFeatures[x], config.outFeatures[x], config.posEnc[x],
                                num_in_features[x], num_out_features[x], x, settings.batch_size, samples);
    
    if (!init_success)
      return false;
  }

  std::cout << "Loaded img gen" << std::endl;
  return true;
}

bool file_exists(const std::string& name) 
{
  ifstream f(name.c_str());
  return f.good();
}



void ImageGenerator::printNetwork(IGUniquePtr<nvinfer1::INetworkDefinition>& network)
{
  for (int k=0; k < network->getNbInputs(); k++)
    std::cout << "Input " << k << " " << network->getInput(k)->getDimensions() << std::endl;

  for (int x = 0; x < network->getNbLayers(); x++)
    std::cout << network->getLayer(x)->getName() << " " << (int )network->getLayer(x)->getType() << std::endl;
 
  for (int k=0; k < network->getNbOutputs(); k++)
    std::cout << "Output " << k << " " << network->getOutput(k)->getDimensions() << std::endl;
}

// TODO: remove hardcoding
bool ImageGenerator::inference(Camera& camera, cudaSurfaceObject_t output_surf, int batch_size, int num_samples,
                               std::vector<FeatureSet*>& feature_sets, std::vector<Encoding>& encodings)
{
  inference_bindings[0][0] = camera.getDeviceFeatureVector();

  if (batch_size <= 0)
  {
    return false;
  }

  if (contexts.size() == 1)
  {
    RayMarchFromPoses* feature_set = (RayMarchFromPoses*)camera.getFeatureSet();

    int c = 0;
    for (int b_start = 0; b_start < settings.total_size; b_start += batch_size)
    {
      CUDA_CHECK;
      camera.UpdateFeaturesBatch(b_start, batch_size);

      contexts[0]->setBindingDimensions(0, Dims3(batch_size, 128, camera.getEncoding()->getNumFeatures()));
      bool status = contexts[0]->executeV2(inference_bindings[0].data());
      if (!status)
      {
        return false;
      }
      CUDA_CHECK;

      copyResultRaymarch(d_outputs[d_outputs.size() - 1], output_surf, settings.width, settings.height,
                         feature_set->getZVals(), camera.getDeviceFeatureVector(), feature_set->getNumRaySamples(),
                         batch_size, b_start);
      c += 1;
    }
  }
  // this is our test setup
  if (contexts.size() == 2)
  {
    if (feature_sets[0]->type == "RayMarchFromPoses")
    {
      RayMarchFromPoses* feature_set = (RayMarchFromPoses*) feature_sets[0];

      int c = 0;
      float cb_fc1 = 0.0f;
      float cb_fc2 = 0.0f;
      float cb_if1 = 0.0f;
      float cb_if2 = 0.0f;
      float cb_rm = 0.0f;

      int n_total_samples = 0;
      
      for (int b_start = 0; b_start < settings.total_size; b_start += batch_size)
      {
        CUDA_CHECK;

        // do the cam pass
        std::chrono::steady_clock::time_point current_bi = std::chrono::steady_clock::now();
        camera.UpdateFeaturesBatch(b_start, batch_size);
        cudaDeviceSynchronize();
        std::chrono::steady_clock::time_point current_ai = std::chrono::steady_clock::now();
        float fc1 = std::chrono::duration_cast<std::chrono::microseconds> (current_ai - current_bi).count();

        contexts[0]->setBindingDimensions(0, Dims2(batch_size, camera.getEncoding()->getNumFeatures()));
        inference_bindings[0][0] = camera.getDeviceFeatureVector();

        current_bi = std::chrono::steady_clock::now();
        bool status = contexts[0]->executeV2(inference_bindings[0].data());
        current_ai = std::chrono::steady_clock::now();
        float inferenc1 = std::chrono::duration_cast<std::chrono::microseconds> (current_ai - current_bi).count();

        if (!status)
        {
          return false;
        }
        CUDA_CHECK;

        // we now have the batch from the camera in output0 .. these are the depths for the second net
        current_bi = std::chrono::steady_clock::now();
        int num_act_inputs =
          feature_set->updateFeatures(camera.getPosition(), camera.getRotMatrix(), camera.getFeaturesClean(), camera.getDeviceFeatureVector(), 
                                      settings.width, settings.height, b_start, batch_size, (float*)d_outputs[0], 128);
        current_ai = std::chrono::steady_clock::now();
        float fc2 = std::chrono::duration_cast<std::chrono::microseconds> (current_ai - current_bi).count();

        contexts[1]->setBindingDimensions(0, Dims2(num_act_inputs, encodings[1].getNumFeatures()));
        n_total_samples += num_act_inputs;

        inference_bindings[1][0] = feature_set->getFeatures();

        current_bi = std::chrono::steady_clock::now();
        status = contexts[1]->executeV2(inference_bindings[1].data());
        current_ai = std::chrono::steady_clock::now();
        float inferenc2 = std::chrono::duration_cast<std::chrono::microseconds> (current_ai - current_bi).count();
        
        if (!status)
        {
          return false;
        }

        current_bi = std::chrono::steady_clock::now();
        feature_set->raymarch(d_outputs[1], output_surf, batch_size, b_start, 
                              settings.width, settings.height, camera.getDeviceFeatureVector());        
        c += 1;
        cudaDeviceSynchronize();
        current_ai = std::chrono::steady_clock::now();
        float rm = std::chrono::duration_cast<std::chrono::microseconds> (current_ai - current_bi).count();
        
        CUDA_CHECK;
        cb_fc1 += fc1;
        cb_fc2 += fc2;
        cb_if1 += inferenc1;
        cb_if2 += inferenc2;
        cb_rm += rm;
      }

      s_inferenc1 += cb_if1 / 1000.0f;
      s_inferenc2 += cb_if2 / 1000.0f;
      s_fc1 += cb_fc1 / 1000.0f;
      s_fc2 += cb_fc2 / 1000.0f;
      s_rm += cb_rm / 1000.0f;
      s_num_total_samples += n_total_samples;

      sample_count++;

      if (sample_count % logging_interval == 0)
      {
        std::cout << "Inference 1:" << s_inferenc1 / logging_interval << ", 2:" << s_inferenc2 / logging_interval 
                  << " | fc1: " << s_fc1 / logging_interval << ", fc2: " << s_fc2 / logging_interval << ", rm: " << s_rm / logging_interval
                  << ", avg samples ppx: " << s_num_total_samples / (float) (logging_interval * settings.total_size) 
                  << " (total: " << s_num_total_samples / logging_interval << ")" 
                  << ", frames: " << sample_count << std::endl;
        
        s_inferenc1 = 0.0f;
        s_inferenc2 = 0.0f;
        s_fc1 = 0.0f;
        s_fc2 = 0.0f;
        s_rm = 0.0f;
        s_num_total_samples = 0;
      } 
    }
    else
    {
      RayMarchFromCoarse* feature_set = (RayMarchFromCoarse*) feature_sets[0];

      int c = 0;
      float cb_fc1 = 0.0f;
      float cb_fc2 = 0.0f;
      float cb_if1 = 0.0f;
      float cb_if2 = 0.0f;
      float cb_rm = 0.0f;

      std::cout << " nerf!" << std::endl;

      for (int b_start = 0; b_start < settings.total_size; b_start += batch_size)
      {
        // do the cam pass
        CUDA_CHECK;
        std::chrono::steady_clock::time_point current_bi = std::chrono::steady_clock::now();
        camera.UpdateFeaturesBatch(b_start, batch_size);
        cudaDeviceSynchronize();
        std::chrono::steady_clock::time_point current_ai = std::chrono::steady_clock::now();
        float fc1 = std::chrono::duration_cast<std::chrono::microseconds> (current_ai - current_bi).count();

        contexts[0]->setBindingDimensions(0, Dims3(batch_size, 64, camera.getEncoding()->getNumFeatures()));
        inference_bindings[0][0] = camera.getDeviceFeatureVector();

        current_bi = std::chrono::steady_clock::now();
        bool status = contexts[0]->executeV2(inference_bindings[0].data());
        current_ai = std::chrono::steady_clock::now();
        float inferenc1 = std::chrono::duration_cast<std::chrono::microseconds> (current_ai - current_bi).count();

        if (!status)
        {
          return false;
        }
        CUDA_CHECK;

        // we now have the batch from the camera in output0 .. these are the depths for the second net
        current_bi = std::chrono::steady_clock::now();
        feature_set->updateFeatures(camera.getPosition(), camera.getRotMatrix(), camera.getFeaturesClean(), camera.getFeaturesClean(), 
                                    settings.width, settings.height, b_start, batch_size, nullptr, (float*)d_outputs[0], 128);
        current_ai = std::chrono::steady_clock::now();
        float fc2 = std::chrono::duration_cast<std::chrono::microseconds> (current_ai - current_bi).count();

        contexts[1]->setBindingDimensions(0, Dims2(batch_size * (128 + 64), encodings[1].getNumFeatures()));
        inference_bindings[1][0] = feature_set->getFeatures();

        current_bi = std::chrono::steady_clock::now();
        status = contexts[1]->executeV2(inference_bindings[1].data());
        current_ai = std::chrono::steady_clock::now();
        float inferenc2 = std::chrono::duration_cast<std::chrono::microseconds> (current_ai - current_bi).count();
        
        if (!status)
        {
          return false;
        }

        current_bi = std::chrono::steady_clock::now();
        copyResultRaymarch(d_outputs[1], output_surf, settings.width, settings.height,
                           feature_set->getZVals(), camera.getDeviceFeatureVector(), 128 + 64, 
                           batch_size, b_start);
        c += 1;
        cudaDeviceSynchronize();
        current_ai = std::chrono::steady_clock::now();
        float rm = std::chrono::duration_cast<std::chrono::microseconds> (current_ai - current_bi).count();
        
        CUDA_CHECK;
        cb_fc1 += fc1;
        cb_fc2 += fc2;
        cb_if1 += inferenc1;
        cb_if2 += inferenc2;
        cb_rm += rm;
        
        std::cout << " Inference 1:" << inferenc1 / 1000.0f << " 2:" << inferenc2 / 1000.0f 
                  << " fc1: " << fc1 / 1000.0f << " fc2: " << fc2 / 1000.0f << " rm: " << rm / 1000.0f << std::endl;
      }

      std::cout << "Sum Inference 1:" << cb_if1 / 1000.0f << " 2:" << cb_if2 / 1000.0f 
                << " fc1: " << cb_fc1 / 1000.0f << " fc2: " << cb_fc2 / 1000.0f << " rm: " << cb_rm / 1000.0f << std::endl;
    }
  }

  return true;
}