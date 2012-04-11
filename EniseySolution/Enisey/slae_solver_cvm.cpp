/** \file test_slae_solver_cvm.h
  Реализация SlaeSolverCVM.*/
#include "slae_solver_cvm.h"
#include <vector>

#include "cvm.h"

void SlaeSolverCVM::Solve(
    std::vector<std::pair<int, int> > const &A_indexes, 
    std::vector<double> const &A_values, 
    std::vector<double> const &B, 
    std::vector<double> *X) {
  int size = B.size();
  cvm::srmatrix A(size);
  auto a_val = A_values.begin();
  for(auto a_index = A_indexes.begin(); a_index != A_indexes.end(); 
      ++a_index) {
    // В CVM нумерация с единицы.
    A(a_index->first + 1, a_index->second + 1) = *a_val;
    ++a_val;
  }
  cvm::rvector B_cvm(size);
  std::copy(B.begin(), B.end(), B_cvm.begin() );
  cvm::rvector X_cvm(size);
  X_cvm.solve(A, B_cvm);
  X->resize(size);
  std::copy(X_cvm.begin(), X_cvm.end(), X->begin() );
}