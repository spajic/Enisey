#pragma once

#include "slae_solver_ice.h"

namespace Enisey {
class SlaeSolverIceCVM : public virtual SlaeSolverIce {
 public:
  virtual void Solve(
      const Enisey::IntSequence &AIndexes,
      const Enisey::DoubleSequence &AValues,
      const Enisey::DoubleSequence &B,
      Enisey::DoubleSequence &X,
      const Ice::Current& current);
  void ActivateSelfInAdapter(const Ice::ObjectAdapterPtr &adapter);
};
typedef IceUtil::Handle<SlaeSolverIceCVM> SlaeSolverIceCVMPtr;

} // Конец namespace Enisey.