/** \file parallel_manager_ice.h
Класс ParallelManagerIce - реалиазация интерфейса ParallelManagerIceI*/
#pragma once

#include "ParallelManagerIce.h"
#include "parallel_manager_pipe_i.h"
#include "passport_pipe.h"
#include "work_params.h"
#include "calculated_params.h"

#include <log4cplus/logger.h>

namespace Enisey {

class ParallelManagerIceServant : public virtual ParallelManagerIce {
public:
  ParallelManagerIceServant();
  ~ParallelManagerIceServant();
  virtual void SetParallelManagerType(
    ParallelManagerType type,
    const ::Ice::Current& /* = ::Ice::Current */
  );
  virtual void TakeUnderControl(
      const ::Enisey::PassportSequence&, 
      const ::Ice::Current& /* = ::Ice::Current */
  );
  
  virtual void SetWorkParams(
        const ::Enisey::WorkParamsSequence&, 
        const ::Ice::Current& /* = ::Ice::Current */
  );
  
  virtual void CalculateAll(const ::Ice::Current& /* = ::Ice::Current */);
  virtual void GetCalculatedParams(
      ::Enisey::CalculatedParamsSequence&, 
      const ::Ice::Current& /* = ::Ice::Current */
  );  
  void ActivateSelfInAdapter(const Ice::ObjectAdapterPtr &adapter);
private:
  log4cplus::Logger log;     

  ParallelManagerPipeI *manager_;
  std::vector<PassportPipe> passports_;
  std::vector<WorkParams> work_params_;
  std::vector<CalculatedParams> calculated_params_;

  // Функции преобразования параметров между ICE и C++.
  void SavePassportSequenceToVector( 
      const ::Enisey::PassportSequence &passport_seq );
  void SaveWorkParamsSequenceToVector( 
      const ::Enisey::WorkParamsSequence &wp_seq );
  void SaveCalculatedParamsVectorToSequence( 
      ::Enisey::CalculatedParamsSequence &cp_seq );
};

typedef // Тип умного указателя на объект ParallelManagerIceServant.
    IceUtil::Handle<ParallelManagerIceServant> ParallelManagerIceServantPtr;
} // Конец namespace Enisey.