/** \file gas_transfer_system_ice_client.h
  Решатель ГТС, обращающийся к серверу ICE.*/
#pragma once
#include "gas_transfer_system_i.h"
#include <vector>
class GasTransferSystemIceClient : public GasTransferSystemI {
 public:
   virtual void PeroformBalancing(
     const std::vector<std::string> &MatrixConnectionsFile,
     const std::vector<std::string> &InOutGRSFile,
     const std::vector<std::string> &PipeLinesFile,
     std::vector<std::string> *ResultFile,
     std::vector<double> *AbsDisbalances,
     std::vector<int> *IntDisbalances);
};