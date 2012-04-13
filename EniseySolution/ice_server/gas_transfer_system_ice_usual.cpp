#include "gas_transfer_system_ice_usual.h"
#include <Ice/Ice.h>
#include "gas_transfer_system.h"

void Enisey::GasTransferSystemIceUsual::PerformBalancing(
    const ::Enisey::StringSequence &MatrixConnectionsFile, 
    const ::Enisey::StringSequence &InOutGRSFile, 
    const ::Enisey::StringSequence &PipeLinesFile, 
    ::Enisey::StringSequence &ResultFile,  
    ::Enisey::DoubleSequence &AbsDisbalances,
    ::Enisey::IntSequence &IntDisbalances,
    const ::Ice::Current& /* = ::Ice::Current */ ) {
  GasTransferSystem gts;
  std::cout << "GasTransferSystemIceUsual::PerformBalancing called" << 
      std::endl;
  gts.PeroformBalancing(
    MatrixConnectionsFile,
    InOutGRSFile, 
    PipeLinesFile, 
    &ResultFile, 
    &AbsDisbalances, 
    &IntDisbalances);
}
void Enisey::GasTransferSystemIceUsual::ActivateSelfInAdapter(
  const Ice::ObjectAdapterPtr &adapter) {
    // Структура Identity требуется для идентификации Servanta в адаптере ASM.
    Ice::Identity id;
    id.name = "GasTransferSystemIceUsual";
    try {
      adapter->add(this, id);
    } catch(const Ice::Exception &ex) {
      std::cout << ex.what();
    }
}