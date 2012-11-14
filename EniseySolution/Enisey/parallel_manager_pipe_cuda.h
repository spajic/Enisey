/** \file parallel_manager_pipe_cuda.h
Менеджер параллельного моделирования труб, использующий CUDA.*/
#pragma once
#include <vector>
#include "parallel_manager_pipe_i.h"

//Forward-declarations:
class ModelPipeSequential;

class ParallelManagerPipeCUDA : public ParallelManagerPipeI {
public:
  virtual void TakeUnderControl(std::vector<PassportPipe> const &passports);
  virtual void SetWorkParams(std::vector<WorkParams> const &work_params);
  virtual void CalculateAll();
  virtual void GetCalculatedParams(std::vector<CalculatedParams> *calculated_params);
private:
  
};