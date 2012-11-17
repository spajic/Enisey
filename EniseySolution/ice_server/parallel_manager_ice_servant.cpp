/** \file parallel_manager_ice.cpp
Класс ParallelManagerIce - реалиазация интерфейса ParallelManagerIceI*/
#include "parallel_manager_ice_servant.h"
#include <Ice/Ice.h>
#include <iostream>

namespace Enisey {
void ParallelManagerIceServant::TakeUnderControl( 
    const ::Enisey::PassportSequence&, 
    const ::Ice::Current& ) {
  std::cout << "ParallelManagerIce::TakeUnderControl" << std::endl;
}
void ParallelManagerIceServant::SetWorkParams(
    const ::Enisey::WorkParamsSequence&, 
    const ::Ice::Current&) {
  std::cout << "ParallelManagerIce::SetWorkParams" << std::endl;
}
void ParallelManagerIceServant::CalculateAll(const ::Ice::Current&) {
  std::cout << "ParallelManagerIce::CalculateAll" << std::endl;
}
void ParallelManagerIceServant::GetCalculatedParams(
  ::Enisey::CalculatedParamsSequence&, 
  const ::Ice::Current&) {
  std::cout << "ParallalManagerIce::GetCalculatedParams" << std::endl;
}
void ParallelManagerIceServant::
    ActivateSelfInAdapter(const Ice::ObjectAdapterPtr &adapter) {
  Ice::Identity id;
  id.name = "ParallelManagerIce";
  try {
    adapter->add(this, id);
  } catch(const Ice::Exception &ex) {
    std::cout << ex.what();
  }
}
} // Конец namespace Enisey.