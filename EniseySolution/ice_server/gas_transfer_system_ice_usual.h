#pragma once

#include "gas_transfer_system_ice.h"

namespace Enisey {

class GasTransferSystemIceUsual : public virtual GasTransferSystemIce {
 public:
  virtual void PerformBalancing(
      const ::Enisey::StringSequence &MatrixConnectionsFile, 
      const ::Enisey::StringSequence &InOutGRSFile, 
      const ::Enisey::StringSequence &PipeLinesFile, 
      ::Enisey::StringSequence &ResultFile,  
      const ::Ice::Current& /* = ::Ice::Current */
  );
};
typedef // Тип умного указателя на объект GasTransferSystemIceUsual.
    IceUtil::Handle<GasTransferSystemIceUsual> GasTransferSystemIceUsualPtr;

} // Конец namespace Enisey.