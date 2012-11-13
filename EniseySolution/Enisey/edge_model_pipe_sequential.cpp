#include "edge_model_pipe_sequential.h"
#include "model_pipe_sequential.h"
#include "passport_pipe.h"

std::string EdgeModelPipeSequential::GetName() {
  return "EdgeModelPipeSequential";
}
void EdgeModelPipeSequential::set_gas_in(const Gas* gas) {
  pointer_to_model_->set_gas_in(gas);
}
void EdgeModelPipeSequential::set_gas_out(const Gas* gas) {
  pointer_to_model_->set_gas_out(gas);
}
double EdgeModelPipeSequential::q() {
  return pointer_to_model_->q();
}
double EdgeModelPipeSequential::dq_dp_in() {
  return pointer_to_model_->dq_dp_in();
}
double EdgeModelPipeSequential::dq_dp_out() {
  return pointer_to_model_->dq_dp_out();
}
bool EdgeModelPipeSequential::IsReverse() {
  return  pointer_to_model_->IsReverse();
}
const Gas& EdgeModelPipeSequential::gas_in() {
  return pointer_to_model_->gas_in();
}
const Gas& EdgeModelPipeSequential::gas_out() {
  return pointer_to_model_->gas_out();
}
Passport* EdgeModelPipeSequential::passport() {
  return pointer_to_model_->passport();
}
