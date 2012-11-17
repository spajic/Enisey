/** \file parallel_manager_pipe_ice.h
Менеджер параллельного моделирования труб, использующий сервис ICE.*/
#pragma once
#include "parallel_manager_pipe_i.h"
#include "ParallelManagerIceI.h"
#include <Ice/Ice.h>

class ParallelManagerPipeIce : public ParallelManagerPipeI {
public:
  ParallelManagerPipeIce();
  ~ParallelManagerPipeIce();
  virtual void TakeUnderControl(std::vector<PassportPipe> const &passports);
  virtual void SetWorkParams(std::vector<WorkParams> const &work_params);
  virtual void CalculateAll();
  virtual void GetCalculatedParams(
      std::vector<CalculatedParams> *calculated_params);
private:
  Ice::CommunicatorPtr ic;
  Enisey::ParallelManagerIceIPrx proxy;
};