/** \file gas_transfer_system_ice_client.h
  Решатель ГТС, обращающийся к серверу ICE.*/
#pragma once
#include "gas_transfer_system_i.h"
#include <vector>
#include <string>
#include <Ice/Ice.h>
#include "gas_transfer_system_ice.h"

class GasTransferSystemIceClient : public GasTransferSystemI {
public:
  GasTransferSystemIceClient();
  GasTransferSystemIceClient(std::string);
  ~GasTransferSystemIceClient();
   virtual void PeroformBalancing(
       const std::vector<std::string> &MatrixConnectionsFile,
       const std::vector<std::string> &InOutGRSFile,
       const std::vector<std::string> &PipeLinesFile,
       std::vector<std::string> *ResultFile,
       std::vector<double> *AbsDisbalances,
       std::vector<int> *IntDisbalances);
private:
  void Init(std::string endpoint);
  Ice::CommunicatorPtr ic_;
  Enisey::GasTransferSystemIcePrx gts_proxy_;
};