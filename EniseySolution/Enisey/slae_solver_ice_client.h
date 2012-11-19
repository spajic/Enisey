/** \file slae_solver_ice_client.h
  Решатель СЛАУ, обращающийся к серверу ICE.*/
#pragma once
#include "slae_solver_i.h"
#include <vector>
#include <string>
#include <Ice/Ice.h>
#include "slae_solver_ice.h"

class SlaeSolverIceClient : public SlaeSolverI {
public:
   SlaeSolverIceClient();
   SlaeSolverIceClient(std::string endpoint);
   ~SlaeSolverIceClient();
  /** Решить СЛАУ AX = B. 
  Предполагаются разреженные СЛАУ, для этого передаётся вектор номеров 
  ненулевых коэффициентов A, в соответствующем векторе A_values - их
  значения.*/
  virtual void Solve(
      std::vector<int> const &A_indexes,
      std::vector<double> const &A_values,
      std::vector<double> const &B,
      std::vector<double> *X);
private:
  Ice::CommunicatorPtr ic_;
  Enisey::SlaeSolverIcePrx solver_proxy_;

  void Prepare(std::string endpoint);
};