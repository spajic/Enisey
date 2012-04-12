/** \file test_slae_solver_cvm.cpp
  Тесты для класса SlaeSolverCVM из slae_solver_cvm.h*/
#include "slae_solver_cvm.h"

#include "gtest/gtest.h"
#include "test_utils.h"

#include <vector>
#include "test_utils.h"

TEST(SlaeSolverCVMTest, SolvesSimpleSlae) {
/* Проверяем решатель системы на базе CVM.cpp
Решаем СЛАУ
[2 0 1]   [1]   [5]
[0 3 0] * [2] = [6]
[1 2 3]   [3]   [14]
Формат передачи параметров:
  A_indexes - индексы (строка, столбец) ненулевых коэф-ов А.
  A_vals - значения этих коэф-ов в соответствующем порядке.
  B - вектор b.
  X - вектор найденного решения. 
Так сделано для эффективности передачи разреженных матриц. 
Хотя для них, конечно, есть свои форматы типа CSR.*/
  SlaeSolverCVM solver;
  std::vector<int> A_indexes = MakeSimpleSlaeAIndexes();
  std::vector<double> A_vals = MakeSimpleSlaeAValues();
  std::vector<double> b = MakeSimpleSlaeB();
  std::vector<double> etalon_x = MakeSimpleSlaeX();
  std::vector<double> x;
  x.reserve( b.size() );

  solver.Solve(A_indexes, A_vals, b, &x);
  EXPECT_TRUE( std::equal( x.begin(), x.end(), etalon_x.begin() ) );
}