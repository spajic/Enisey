/** \file test_slae_solver_cusp.h
  Решатель СЛАУ на базе cusp.*/
#pragma once

#include "slae_solver_i.h"

#include <vector>

class SlaeSolverCusp : public SlaeSolverI {
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
