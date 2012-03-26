#include "passport_pipe.h"

PassportPipe::PassportPipe() : length_(-1), d_outer_(-1), d_inner_(-1),
    p_max_(-1), p_min_(-1), hydraulic_efficiency_coeff_(-1), roughness_coeff_(-1),
    heat_exchange_coeff_(-1), t_env_(-1) {}
std::string PassportPipe::GetName() {
  return "PassportPipe";
}