/** \file parallel_manager_ice.h
Класс ParallelManagerIce - реалиазация интерфейса ParallelManagerIceI*/
#pragma once

#include "ParallelManagerIceI.h"

namespace Enisey {

  class ParallelManagerIceServant : public virtual ParallelManagerIceI {
  public:
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
  };
  typedef // Тип умного указателя на объект ParallelManagerIceServant.
    IceUtil::Handle<ParallelManagerIceServant> ParallelManagerIceServantPtr;

} // Конец namespace Enisey.