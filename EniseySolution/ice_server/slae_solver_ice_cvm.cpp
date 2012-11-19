#include "slae_solver_ice_cvm.h"
#include <Ice/Ice.h>
#include "slae_solver_cvm.h"

#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>

Enisey::SlaeSolverIceCVM::SlaeSolverIceCVM() {
  log = log4cplus::Logger::getInstance( 
      LOG4CPLUS_TEXT("IceServer.SlaeServant") );
}

void Enisey::SlaeSolverIceCVM::Solve(
    const Enisey::IntSequence &AIndexes, 
    const Enisey::DoubleSequence &AValues, 
    const Enisey::DoubleSequence &B, 
    Enisey::DoubleSequence& X,
    const Ice::Current& current) {
LOG4CPLUS_INFO(log, 
    "Start Solve; Size = " << B.size() << "; Solver: hard-wired to CVM");
  SlaeSolverCVM slae_solver_cvm;
  slae_solver_cvm.Solve(AIndexes, AValues, B, &X);
LOG4CPLUS_INFO(log, "End Solve");
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
