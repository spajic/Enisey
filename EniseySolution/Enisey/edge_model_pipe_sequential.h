#pragma once

#include <string> 

#include "edge.h"

// forward-declaration
class ModelPipeSequential;
struct Gas;
struct Passport;

class EdgeModelPipeSequential: public Edge
{
public:
  void set_gas_in(const Gas* gas);
  void set_gas_out(const Gas* gas);
  const Gas& gas_in();
  const Gas& gas_out();
  std::string GetName();
  double q();
  double dq_dp_in();
  double dq_dp_out();
  bool IsReverse();
  Passport *passport();
  ModelPipeSequential* pointer_to_model_;
};