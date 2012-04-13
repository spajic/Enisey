#include "slae_solver_ice_cvm.h"
#include <Ice/Ice.h>
#include "slae_solver_cvm.h"

void Enisey::SlaeSolverIceCVM::Solve(
    const Enisey::IntSequence &AIndexes, 
    const Enisey::DoubleSequence &AValues, 
    const Enisey::DoubleSequence &B, 
    Enisey::DoubleSequence& X,
    const Ice::Current& current) {
  std::cout << "Execute Enisey::SlaeSloverCVM::Solve\n";
  SlaeSolverCVM slae_solver_cvm;
  slae_solver_cvm.Solve(AIndexes, AValues, B, &X);
}
void Enisey::SlaeSolverIceCVM::ActivateSelfInAdapter(
    const Ice::ObjectAdapterPtr &adapter) {
  // Структура Identity требуется для идентификации Servanta в адаптере ASM.
  Ice::Identity id;
  id.name = "SlaeSolverIceCVM";
  try {
    adapter->add(this, id);
  } catch(const Ice::Exception &ex) {
    std::cout << ex.what();
  }
}
