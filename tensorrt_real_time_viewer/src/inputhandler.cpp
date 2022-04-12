#include "inputhandler.h"

#include "GL/platform/Application.h"

#include "../include/neuralrenderer.h"
#include "../include/camera.h"

#include <iostream>

InputHandler::InputHandler(NeuralRenderer& renderer,Camera& camera)
  : renderer(renderer), camera(camera), mouse_left(false)
{
  std::cout << std::endl;
  std::cout << "Hotkeys:" << std::endl
            << "  <W,A,S,D> Movement" << std::endl;
  std::cout << std::endl;
}

void InputHandler::keyDown(GL::platform::Key key)
{
  switch (key)
  {
  case GL::platform::Key::C_W:
    camera.MovementKeyPressed('w');
    break;
  case GL::platform::Key::C_A:
    camera.MovementKeyPressed('a');
    break;
  case GL::platform::Key::C_S:
    camera.MovementKeyPressed('s');
    break;
  case GL::platform::Key::C_D:
    camera.MovementKeyPressed('d');
    break;
  case GL::platform::Key::C_Q:
      camera.MovementKeyPressed('q');
      break;
  case GL::platform::Key::C_E:
      camera.MovementKeyPressed('e');
      break;
  default:
    break;
  }
}

void InputHandler::keyUp(GL::platform::Key key)
{
  switch (key)
  {
  case GL::platform::Key::C_W:
      camera.MovementKeyReleased('w');
    break;
  case GL::platform::Key::C_A:
      camera.MovementKeyReleased('a');
    break;
  case GL::platform::Key::C_S:
      camera.MovementKeyReleased('s');
    break;
  case GL::platform::Key::C_D:
      camera.MovementKeyReleased('d');
    break;
  case GL::platform::Key::C_Q:
      camera.MovementKeyReleased('q');
      break;
  case GL::platform::Key::C_E:
      camera.MovementKeyReleased('e');
      break;
  case GL::platform::Key::BACKSPACE:
    break;
  case GL::platform::Key::C_F:
    break;
  case GL::platform::Key::ESCAPE:
    GL::platform::quit();
    break;
  default:
    break;
  }
}

void InputHandler::buttonDown([[maybe_unused]]GL::platform::Button button, [[maybe_unused]]int x, [[maybe_unused]]int y)
{
  if (button == GL::platform::Button::LEFT)
  {
    mouse_left = true;
  }
  last_mouse_pos = glm::ivec2(x, y);
}

void InputHandler::buttonUp([[maybe_unused]]GL::platform::Button button, [[maybe_unused]]int x, [[maybe_unused]]int y)
{
  if (button == GL::platform::Button::LEFT)
  {
    mouse_left = false;
  }
}

void InputHandler::mouseMove([[maybe_unused]]int x, [[maybe_unused]]int y)
{
  glm::ivec2 mouse_pos(x, y);
  if (mouse_left)
    camera.MouseDrag(mouse_pos - last_mouse_pos);
  last_mouse_pos = mouse_pos;
}

void InputHandler::mouseWheel([[maybe_unused]]int delta)
{
}
