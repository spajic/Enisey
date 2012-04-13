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
typedef // ��� ������ ��������� �� ������ GasTransferSystemIceUsual.
    IceUtil::Handle<GasTransferSystemIceUsual> GasTransferSystemIceUsualPtr;

} // ����� namespace Enisey.