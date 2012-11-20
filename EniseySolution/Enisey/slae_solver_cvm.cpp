/** \file test_slae_solver_cvm.cpp
  Реализация SlaeSolverCVM.*/
#include "slae_solver_cvm.h"
#include <vector>

#include "cvm.h"
#include <math.h>

void SlaeSolverCVM::Solve(
    std::vector<int> const &A_indexes, 
    std::vector<double> const &A_values, 
    std::vector<double> const &B, 
    std::vector<double> *X) {
  // Длина вектора B = Длина строки матрицы A.
  int size = B.size();
  cvm::srmatrix A(size);
  auto a_val = A_values.begin();
  for(auto a_index = A_indexes.begin(); a_index != A_indexes.end(); 
      ++a_index) {
    // Заполняем матрицу A CVM - нумерация с единицы.
    int row = *a_index / size;
    int col = *a_index - row*size;
    A(row + 1, col + 1) = *a_val;
    ++a_val;
  }
  cvm::rvector B_cvm(size);
  std::copy(B.begin(), B.end(), B_cvm.begin() );
  cvm::rvector X_cvm(size);
  X_cvm.solve(A, B_cvm);
  X->resize(size);
  std::copy(X_cvm.begin(), X_cvm.end(), X->begin() );
}