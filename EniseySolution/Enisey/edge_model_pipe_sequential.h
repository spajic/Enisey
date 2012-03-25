#pragma once

#include <string> 

#include "edge.h"

// forward-declaration
class ModelPipeSequential;
struct Gas;

class EdgeModelPipeSequential: public Edge
{
public:
  void set_gas_in(const Gas* gas);
  void set_gas_out(const Gas* gas);
  std::string GetName();
  float q();
  ModelPipeSequential* pointer_to_model_;
};