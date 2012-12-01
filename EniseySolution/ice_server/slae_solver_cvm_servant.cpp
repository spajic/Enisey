#include "slae_solver_cvm_servant.h"
#include <Ice/Ice.h>
#include "slae_solver_cvm.h"
#include "slae_solver_cusp.cuh"

#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>

#include <string>

#include <stdexcept>

namespace Enisey {

typedef std::unique_ptr<SlaeSolverI> SlaeSolverIPtr;

using namespace log4cplus;

SlaeSolverIceCVM::SlaeSolverIceCVM() {
  log = Logger::getInstance(LOG4CPLUS_TEXT("IceServer.SlaeServant"));
  solver_type_ = "CVM"; // По умолчанию - CVM.
}

void SlaeSolverIceCVM::SetSolverType(
    const std::string &solver_type,
    const Ice::Current& current) {
  solver_type_ = solver_type;
}

void SlaeSolverIceCVM::Solve(
    const IntSequence &AIndexes, 
    const DoubleSequence &AValues, 
    const DoubleSequence &B, 
    Enisey::DoubleSequence& X,
    const Ice::Current& current) {
LOG4CPLUS_INFO(log, 
    "Start Solve; Size = " << B.size() << "; Solver:" << solver_type_.c_str() );
  SlaeSolverIPtr slae_solver;
  if(solver_type_ == "CVM" ) {
    slae_solver = SlaeSolverIPtr(new SlaeSolverCVM);
  }
  else if(solver_type_ == "CUSP") {
    slae_solver = SlaeSolverIPtr(new SlaeSolverCusp);
  }   
  else {
    throw std::invalid_argument("Invalid solver type: " + solver_type_);
  }
  slae_solver->Solve(AIndexes, AValues, B, &X);
LOG4CPLUS_INFO(log, "End Solve");
}

void SlaeSolverIceCVM::ActivateSelfInAdapter(
    const Ice::ObjectAdapterPtr &adapter) {
  // Структура Identity требуется для идентификации Servanta в адаптере ASM.
  Ice::Identity id;
  id.name = "SlaeSolverIce";
  try {
    adapter->add(this, id);
  } catch(const Ice::Exception &ex) {
    std::cout << ex.what();
  }
}

} // Конец namespace Enisey.