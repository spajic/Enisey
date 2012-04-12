/** \file test_slae_solver_cvm.cpp
  Решатель СЛАУ на базе CVM.*/
#pragma once
#include "slae_solver_i.h"
#include <vector>
class SlaeSolverCVM : public SlaeSolverI {
 public:
  /** Решить СЛАУ AX = B. 
  Предполагаются разреженные СЛАУ, для этого передаётся вектор номеров 
  ненулевых коэффициентов A, в соответствующем векторе A_values - их
  значения.*/
  virtual void Solve(
      std::vector<std::pair<int, int> > const &A_indexes,
      std::vector<double> const &A_values,
      std::vector<double> const &B,
      std::vector<double> *X);
};