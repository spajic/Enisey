/** \file gas_transfer_system_ice_client.cpp
  Реализация GasTransferSystemIceClient.*/
#include "gas_transfer_system_ice_client.h"
#include "Ice/Ice.h"
#include "gas_transfer_system_ice.h"
#include <vector>

GasTransferSystemIceClient::GasTransferSystemIceClient() {
  Init("GasTransferSystemIceUsual:default -p 10000");
}

GasTransferSystemIceClient::GasTransferSystemIceClient(std::string endpoint) {
  Init(endpoint);
}

GasTransferSystemIceClient::~GasTransferSystemIceClient() {
  if(ic_) {
    try {
      ic_->destroy();
    } 
    catch(const Ice::Exception &e) {
        std::cerr << e << std::endl;
    }
  }
}


void GasTransferSystemIceClient::Init(std::string endpoint) {
  try {
    ic_ = Ice::initialize();
    Ice::ObjectPrx base = ic_->stringToProxy(endpoint);
    gts_proxy_ = Enisey::GasTransferSystemIcePrx::checkedCast(base);
    if(!gts_proxy_) {
      std::cout << "Invalid proxy";
    }  
  } 
  catch(const Ice::Exception &ex){  
    std::cerr << ex.what() << std::endl;
  } 
}


void GasTransferSystemIceClient::PeroformBalancing(
    const std::vector<std::string> &MatrixConnectionsFile, 
    const std::vector<std::string> &InOutGRSFile, 
    const std::vector<std::string> &PipeLinesFile, 
    std::vector<std::string> *ResultFile, 
    std::vector<double> *AbsDisbalances, 
    std::vector<int> *IntDisbalances) {  
  try {     
    gts_proxy_->PerformBalancing(
        MatrixConnectionsFile, 
        InOutGRSFile, 
        PipeLinesFile, 
        *ResultFile, 
        *AbsDisbalances, 
        *IntDisbalances);    
  }
  catch(const Ice::Exception &ex){  
    std::cerr << ex.what() << std::endl;
  }
}
