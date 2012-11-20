#pragma once

#include "gas_transfer_system_ice.h"

#include <log4cplus/logger.h>
using log4cplus::Logger;

namespace Enisey {

class GasTransferSystemIceUsual : public virtual GasTransferSystemIce {
  
public:
  GasTransferSystemIceUsual();
  virtual void PerformBalancing(
      const ::Enisey::StringSequence &MatrixConnectionsFile, 
      const ::Enisey::StringSequence &InOutGRSFile, 
      const ::Enisey::StringSequence &PipeLinesFile, 
      ::Enisey::StringSequence &ResultFile,
      ::Enisey::DoubleSequence &AbsDisbalances,
      ::Enisey::IntSequence &IntDisbalances,
      const ::Ice::Current& /* = ::Ice::Current */
  );
  void ActivateSelfInAdapter(const Ice::ObjectAdapterPtr &adapter);
private:
  Logger log_;  
};

typedef // Тип умного указателя на объект GasTransferSystemIceUsual.
IceUtil::Handle<GasTransferSystemIceUsual> GasTransferSystemIceUsualPtr;

} // Конец namespace Enisey.