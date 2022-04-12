#include <iostream>

#include "../include/camera.h"
#include "../include/config.h"
#include "../include/inputhandler.h"
#include "../include/helper.h"
#include "../include/neuralrenderer.h"
#include "../include/settings.h"

#include "GL/platform/Window.h"
#include "GL/platform/Application.h"
#include "argparse/argparse.hpp"


int main(int argc, char* argv[])
{
  argparse::ArgumentParser program("donerf");

  program.add_argument("modelPath")
    .help("Full path to model and config.ini folder")
    .default_value(std::string("sample") + PATH_SEPARATOR);

  program.add_argument("-s", "--size")
    .help("width and height of generated image")
    .nargs(2)
    .default_value(std::vector<unsigned int>{800, 800})
    .scan<'u', unsigned int>();
  program.add_argument("-ws", "--windowSize")
    .help("width and height of window (default is argument --size)")
    .nargs(2)
    .default_value(std::vector<unsigned int>{800, 800})
    .scan<'u', unsigned int>();

  program.add_argument("-bs", "--batchSize")
    .help("size of the batches for inference (max = -1 = width*height)")
    .default_value(-1)
    .scan<'i', int>();
  program.add_argument("-nb", "--numberOfBatches")
    .help("number of batches (overwritten if batchSize set)")
    .default_value((unsigned int) 1)
    .scan<'u', unsigned int>();

  program.add_argument("-w", "--writeImages")
    .help("write images to model directory")
    .default_value(false)
    .implicit_value(true);
  program.add_argument("-d", "--debug")
    .help("debug mode. Will not render to window (interop buffer problems with NSightCompute)")
    .default_value(false)
    .implicit_value(true);


  Settings settings;
  try
  {
    program.parse_args(argc, argv);
    settings.init(program);
  }
  catch (const std::runtime_error& err)
  {
    std::cout << "Argument parsing failed!" << std::endl;
    std::cout << err.what() << std::endl;
    std::cout << program;
    return -1;
  }
  catch (const std::invalid_argument& err)
  {
    std::cout << "Reading parsed arguments failed!" << std::endl;
    std::cout << err.what() << std::endl;
    std::cout << program;
    return -1;
  }

  Config config;
  if (!config.load(settings.model_path))
    return -1;


  Camera camera(settings, config);
  GL::platform::Window window("nr-real-time", settings.window_width, settings.window_height, 0, 0, false);
  NeuralRenderer neural_renderer(settings, config, window, camera, 4, 6);

	InputHandler input_handler(neural_renderer, camera);

  if (!neural_renderer.init())
  {
    std::cout << "NeuralRenderer failed to initialize" << std::endl;
    return -1;
  }

	window.attach(static_cast<GL::platform::DisplayHandler*>(&neural_renderer));
  window.attach(static_cast<GL::platform::KeyboardInputHandler*>(&input_handler));
	window.attach(static_cast<GL::platform::MouseInputHandler*>(&input_handler));
	
  std::cout << "Starting" << std::endl;
	GL::platform::run(neural_renderer);
  return 0;
}