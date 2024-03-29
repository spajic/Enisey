#include "gas_transfer_system_ice_usual.h"
#include <Ice/Ice.h>
#include "GasTransferSystemIce.h"
#include "gas_transfer_system.h"

#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>
using log4cplus::Logger;

namespace Enisey {

GasTransferSystemIceUsual::GasTransferSystemIceUsual() {
  log_ = Logger::getInstance( LOG4CPLUS_TEXT("IceServer.GtsServant") );
  number_of_iterations_ = 1;
}

void GasTransferSystemIceUsual::SetNumberOfIterations(
    int number_of_iterations,
    const ::Ice::Current& /* = ::Ice::Current */) {
  number_of_iterations_ = number_of_iterations;
}

void GasTransferSystemIceUsual::PerformBalancing(
    const ::Enisey::StringSequence &MatrixConnectionsFile, 
    const ::Enisey::StringSequence &InOutGRSFile, 
    const ::Enisey::StringSequence &PipeLinesFile, 
    ::Enisey::StringSequence &ResultFile,  
    ::Enisey::DoubleSequence &AbsDisbalances,
    ::Enisey::IntSequence &IntDisbalances,
    const ::Ice::Current& /* = ::Ice::Current */ ) {
  
LOG4CPLUS_INFO(log_, "PerformBalancing start");  
LOG4CPLUS_INFO(log_, 
    "--Going to Perform balancing " << number_of_iterations_ << " times.");
  for(int i = 0; i < number_of_iterations_; ++i) {
    GasTransferSystem gts;    
    ResultFile.clear();
    AbsDisbalances.clear();
    IntDisbalances.clear();
LOG4CPLUS_INFO(log_, "----Call gts.PeroformBalancing");
    gts.PeroformBalancing(
        MatrixConnectionsFile,
        InOutGRSFile, 
        PipeLinesFile, 
        &ResultFile, 
        &AbsDisbalances, 
        &IntDisbalances);
LOG4CPLUS_INFO(log_, "----Return from gts.PeroformBalancing");
  }
LOG4CPLUS_INFO(log_, "PerformBalancing finish");
}

void GasTransferSystemIceUsual::ActivateSelfInAdapter(
  const Ice::ObjectAdapterPtr &adapter) {
    // ��������� Identity ��������� ��� ������������� Servanta � �������� ASM.
    Ice::Identity id;
    id.name = "GasTransferSystemIceUsual";
    try {
      adapter->add(this, id);
    } catch(const Ice::Exception &ex) {
      std::cout << ex.what();
    }
}

} // ����� namespace Enisey.