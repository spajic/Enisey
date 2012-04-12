/** \file slae_solver_i.cpp
��������� �������� ���� SlaeSolverI. */
#pragma once
#include <vector>
class SlaeSolverI {
 public:
   virtual void Solve(
     std::vector<int> const &A_indexes, 
     std::vector<double> const &A_values, 
     std::vector<double> const &B, 
     std::vector<double> *X) = 0;
};