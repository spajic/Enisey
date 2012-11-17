/** \file parallel_manager_pipe_singlecore.h
Менеджер параллельного моделирования труб, использующий одно ядро ЦП.*/
#pragma once
#include <vector>
#include "parallel_manager_pipe_i.h"
#include "model_pipe_sequential.h"

//Forward-declarations:
//class ModelPipeSequential;

class ParallelManagerPipeSingleCore : public ParallelManagerPipeI{
public:
 virtual void TakeUnderControl(std::vector<PassportPipe> const &passports);
 virtual void SetWorkParams(std::vector<WorkParams> const &work_params);
 virtual void CalculateAll();
 virtual void GetCalculatedParams(std::vector<CalculatedParams> *calculated_params);
private:
  std::vector<ModelPipeSequential> models_;
};