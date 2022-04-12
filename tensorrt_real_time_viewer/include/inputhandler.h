
#ifndef NRREALTIME_INPUTHANDLER_H
#define NRREALTIME_INPUTHANDLER_H

#pragma once

#include <glm/vec2.hpp> 

#include "GL/platform/InputHandler.h"


class NeuralRenderer;
class Camera;

class InputHandler : public virtual GL::platform::KeyboardInputHandler, public virtual GL::platform::MouseInputHandler
{
public:
  InputHandler(NeuralRenderer& renderer, Camera& camera);
  InputHandler() = default;

  InputHandler(const InputHandler&) = delete;
  InputHandler& operator=(const InputHandler&) = delete;

  void keyDown(GL::platform::Key key);
  void keyUp(GL::platform::Key key);
  void buttonDown(GL::platform::Button button, int x, int y);
  void buttonUp(GL::platform::Button button, int x, int y);
  void mouseMove(int x, int y);
  void mouseWheel(int delta);

private:
  NeuralRenderer& renderer;
  Camera& camera;

  bool mouse_left;
  glm::ivec2 last_mouse_pos;
};


#endif