/** \file parallel_manager_pipe_cuda.h
Менеджер параллельного моделирования труб, использующий CUDA.*/
#pragma once
#include "parallel_manager_pipe_i.h"

#include <vector>

//Forward-declarations:
class ModelPipeSequential;
class double2;
class double4;

class ParallelManagerPipeCUDA : public ParallelManagerPipeI {
public:
  ParallelManagerPipeCUDA();
  virtual void TakeUnderControl(std::vector<PassportPipe> const &passports);
  virtual void SetWorkParams(std::vector<WorkParams> const &work_params);
  virtual void CalculateAll();
  virtual void GetCalculatedParams(std::vector<CalculatedParams> *calculated_params);
private:  
  void AllocateMemoryOnDevice();

  int number_of_pipes;

  double* length_dev_;
  double2* d_in_out_dev_;
  double4* hydr_rough_env_exch_dev_;
  double2* p_in_and_t_in_dev_;
  double* p_target_dev_;
  double* q_result_dev_;
  double* den_sc_dev_;
  double* co2_dev_;
  double* n2_dev_;

  // Пасспортные параметры трубы
  double* length_;
  double2* d_in_out_;
  double4* hydr_rough_env_exch_;

  // Рабочие параметры
  double* den_sc_;
  double* co2_; 
  double* n2_;
  //double2* p_in_and_t_in_;
  double* p_target_;
  double* q_result_;
};