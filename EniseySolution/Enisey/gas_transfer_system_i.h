/** \file gas_transfer_system_i.h
Класс GasTransferSystemI - абстракнтый интерфей ГТС.*/
#pragma once
#include <string>
#include <vector>
#include <map>
// Forward-declarations.
class GraphBoost;
class SlaeSolverI;
class ManagerEdge;

/** Класс представляет абстрактный интерфейс ГТС.*/
class GasTransferSystemI {
 public:
   /// Выполнить балансировку системы.
   virtual void PeroformBalancing(
     const std::vector<std::string> &MatrixConnectionsFile,
     const std::vector<std::string> &InOutGRSFile,
     const std::vector<std::string> &PipeLinesFile,
     std::vector<std::string> *ResultFile,
     std::vector<double> *AbsDisbalances,
     std::vector<int> *IntDisbalances
  ) = 0;
};
