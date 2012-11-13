/** \file parallel_manager_pipe_openmp.cpp
Реализация менеджера параллельного моделирования труб, 
использующего OpenMP.*/
#include "parallel_manager_pipe_openmp.h"
#include "model_pipe_sequential.h"
#include <vector>

void ParallelManagerPipeOpenMP::
  TakeUnderControl(std::vector<PassportPipe> const &passports) {
    for(auto p = passports.begin(); p != passports.end(); ++p) {
      PassportPipe passport = *p;
      ModelPipeSequential model(&passport);
      models_.push_back(model);
    }      
}
void ParallelManagerPipeOpenMP::
  SetWorkParams(std::vector<WorkParams> const &work_params) {
    auto wp = work_params.begin();
    for(auto m = models_.begin(); m != models_.end(); ++m) {
      Gas gas_in;
      gas_in.composition.density_std_cond = wp->den_sc_in();
      gas_in.composition.co2 = wp->co2_in();
      gas_in.composition.n2 = wp->n2_in();
      gas_in.work_parameters.p = wp->p_in();
      gas_in.work_parameters.t = wp->t_in();
      m->set_gas_in(&gas_in);

      Gas gas_out;
      gas_out.work_parameters.p = wp->p_out();
      m->set_gas_out(&gas_out);
      ++wp;
    }

}
void ParallelManagerPipeOpenMP::CalculateAll() {
#pragma omp parallel for 
  for(int i = 0; i < models_.size(); ++i) {
    models_[i].Count();
  }
}
void ParallelManagerPipeOpenMP::
  GetCalculatedParams(std::vector<CalculatedParams> *calculated_params){
    calculated_params->erase(calculated_params->begin(), calculated_params->end());
    for(auto m = models_.begin(); m != models_.end(); ++m) {
      CalculatedParams cp;
      cp.set_q         ( m->q() );
      cp.set_dq_dp_in  ( m->dq_dp_in() );
      cp.set_dq_dp_out ( m->dq_dp_out() );
      cp.set_t_out     ( m->gas_out().work_parameters.t );  
      calculated_params->push_back(cp);
    }
}
