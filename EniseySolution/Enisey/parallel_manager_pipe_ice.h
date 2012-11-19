/** \file parallel_manager_pipe_ice.h
Менеджер параллельного моделирования труб, использующий сервис ICE.*/
#pragma once
#include "parallel_manager_pipe_i.h"
#include "ParallelManagerIceI.h"
#include <Ice/Ice.h>

class ParallelManagerPipeIce : public ParallelManagerPipeI {
public:
  ParallelManagerPipeIce();
  ParallelManagerPipeIce(std::string endpoint);
  ~ParallelManagerPipeIce();
  void SetParallelManagerType(std::string);
  virtual void TakeUnderControl(std::vector<PassportPipe> const &passports);
  virtual void SetWorkParams(std::vector<WorkParams> const &work_params);
  virtual void CalculateAll();
  virtual void GetCalculatedParams(
      std::vector<CalculatedParams> *calculated_params);

private:
  Ice::CommunicatorPtr ic;
  Enisey::ParallelManagerIceIPrx proxy;

  void ConvertPassportVecToSequence( 
      std::vector<PassportPipe> const &passports,
      Enisey::PassportSequence *passport_seq
  );
  void ConvertWorkParamsVecToSequence( 
      std::vector<WorkParams> const &work_params,
      Enisey::WorkParamsSequence *wp_seq
  );
  void ConvertCalculatedParamsSeqToVector( 
      Enisey::CalculatedParamsSequence &cp_seq, 
      std::vector<CalculatedParams> * calculated_params      
  );
};