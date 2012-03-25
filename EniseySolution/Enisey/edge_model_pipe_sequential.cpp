#include "edge_model_pipe_sequential.h"

#include "model_pipe_sequential.h"

std::string EdgeModelPipeSequential::GetName()
{
  return "EdgeModelPipeSequential";
}

void EdgeModelPipeSequential::set_gas_in(const Gas* gas)
{
  pointer_to_model_->set_gas_in(gas);
}

void EdgeModelPipeSequential::set_gas_out(const Gas* gas)
{
  pointer_to_model_->set_gas_out(gas);
}

float EdgeModelPipeSequential::q()
{
  return pointer_to_model_->q();
}
