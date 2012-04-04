#include "edge_model_pipe_sequential.h"
#include "model_pipe_sequential.h"

std::string EdgeModelPipeSequential::GetName() {
  return "EdgeModelPipeSequential";
}
void EdgeModelPipeSequential::set_gas_in(const Gas* gas) {
  pointer_to_model_->set_gas_in(gas);
}
void EdgeModelPipeSequential::set_gas_out(const Gas* gas) {
  pointer_to_model_->set_gas_out(gas);
}
float EdgeModelPipeSequential::q() {
  return pointer_to_model_->q();
}
float EdgeModelPipeSequential::dq_dp_in() {
  return pointer_to_model_->dq_dp_in();
}
float EdgeModelPipeSequential::dq_dp_out() {
  return pointer_to_model_->dq_dp_out();
}
float EdgeModelPipeSequential::IsReverse() {
  return  pointer_to_model_->IsReverse();
}
const Gas& EdgeModelPipeSequential::gas_in() {
  return pointer_to_model_->gas_in();
}
const Gas& EdgeModelPipeSequential::gas_out() {
  return pointer_to_model_->gas_out();
}
