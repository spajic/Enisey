#pragma once

#include "SlaeSolverIce.h"

#include <log4cplus/logger.h>

#include <string>

namespace Enisey {
class SlaeSolverIceCVM : public virtual SlaeSolverIce {
public:
  SlaeSolverIceCVM();
  virtual void SetSolverType(
      const std::string  &solver_type, 
      const Ice::Current &current
  );
  virtual void Solve(
      const Enisey::IntSequence &AIndexes,
      const Enisey::DoubleSequence &AValues,
      const Enisey::DoubleSequence &B,
      Enisey::DoubleSequence &X,
      const Ice::Current& current);
  void ActivateSelfInAdapter(const Ice::ObjectAdapterPtr &adapter);
private:
  log4cplus::Logger log;
  std::string solver_type_;
};
typedef IceUtil::Handle<SlaeSolverIceCVM> SlaeSolverIceCVMPtr;

} // Конец namespace Enisey.