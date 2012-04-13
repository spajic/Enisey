#include "gas_transfer_system_ice_usual.h"
#include <Ice/Ice.h>
#include "gas_transfer_system.h"

void Enisey::GasTransferSystemIceUsual::PerformBalancing(
    const ::Enisey::StringSequence &MatrixConnectionsFile, 
    const ::Enisey::StringSequence &InOutGRSFile, 
    const ::Enisey::StringSequence &PipeLinesFile, 
    ::Enisey::StringSequence &ResultFile,  
    const ::Ice::Current& /* = ::Ice::Current */ ) {
  GasTransferSystem gts;
}
