/** \file gas_transfer_system_ice_client.cpp
  Реализация GasTransferSystemIceClient.*/
#include "gas_transfer_system_ice_client.h"
#include "Ice/Ice.h"
#include "gas_transfer_system_ice.h"
#include <vector>

void GasTransferSystemIceClient::PeroformBalancing(
    const std::vector<std::string> &MatrixConnectionsFile, 
    const std::vector<std::string> &InOutGRSFile, 
    const std::vector<std::string> &PipeLinesFile, 
    std::vector<std::string> *ResultFile, 
    std::vector<double> *AbsDisbalances, 
    std::vector<int> *IntDisbalances) {
  Ice::CommunicatorPtr ic;
  try {
    ic = Ice::initialize();
    Ice::ObjectPrx base = ic->stringToProxy("GasTransferSystemIceUsual:default -p 10000");
    Enisey::GasTransferSystemIcePrx gts_proxy = 
      Enisey::GasTransferSystemIcePrx::checkedCast(base);
    if(!gts_proxy) {
      std::cout << "Invalid proxy";
    } else {
      gts_proxy->PerformBalancing(
          MatrixConnectionsFile, 
          InOutGRSFile, 
          PipeLinesFile, 
          *ResultFile, 
          *AbsDisbalances, 
          *IntDisbalances);
    }
  } catch(const Ice::Exception &ex){  
    std::cerr << ex << std::endl;
  } catch(const char *msg) {
    std::cerr << msg << std::endl;
  }
  if(ic) {
    try {
      ic->destroy();
    } catch(const Ice::Exception &e) {
      std::cerr << e << std::endl;
    }
  }
}
