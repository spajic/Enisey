/** \file parallel_manager_pipe_i.h
Абстрактный интерфейс менеджера параллельного моделирования труб.*/
#pragma once
#include <vector>
#include "passport_pipe.h"
#include "work_params.h"
#include "calculated_params.h"

class ParallelManagerPipeI {
public:
  virtual void TakeUnderControl(std::vector<PassportPipe> const &passports) = 0;
  virtual void SetWorkParams(std::vector<WorkParams> const &work_params) = 0;
  virtual void CalculateAll() = 0;
  virtual void GetCalculatedParams(
      std::vector<CalculatedParams> *calculated_params) = 0;
};