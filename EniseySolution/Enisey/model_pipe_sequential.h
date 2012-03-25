#pragma once
#include "gas.h"
#include "passport_pipe.h"

class ModelPipeSequential
{
public:
  ModelPipeSequential();
  ModelPipeSequential(const Passport* passport);
  void set_gas_in(const Gas* gas);
  void set_gas_out(const Gas* gas);
  void Count();
  float q();
private:
  void DetermineDirectionOfFlow();

  float q_;
  bool direction_is_forward_;
  PassportPipe passport_;
  Gas gas_in_;
  Gas gas_out_;
};