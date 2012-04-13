/** \file slae_solver_ice_client.h
  Решатель СЛАУ, обращающийся к серверу ICE.*/
#pragma once
#include "slae_solver_i.h"
#include <vector>
class SlaeSolverIceClient : public SlaeSolverI {
 public:
  /** Решить СЛАУ AX = B. 
  Предполагаются разреженные СЛАУ, для этого передаётся вектор номеров 
  ненулевых коэффициентов A, в соответствующем векторе A_values - их
  значения.*/
  virtual void Solve(
      std::vector<int> const &A_indexes,
      std::vector<double> const &A_values,
      std::vector<double> const &B,
      std::vector<double> *X);
};