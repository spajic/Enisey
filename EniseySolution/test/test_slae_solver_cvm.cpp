/** \file test_slae_solver_cvm.cpp
  Тесты для класса SlaeSolverCVM из slae_solver_cvm.h*/
#include "slae_solver_cvm.h"

#include "gtest/gtest.h"
#include "test_utils.h"

#include <vector>

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
  std::vector<std::pair<int,int> > A_indexes;
  std::vector<double> A_vals;
  A_indexes.push_back( std::make_pair(0, 0) );  A_vals.push_back( 2 );
  A_indexes.push_back( std::make_pair(0, 2) );  A_vals.push_back( 1 );
  A_indexes.push_back( std::make_pair(1, 1) );  A_vals.push_back( 3 );
  A_indexes.push_back( std::make_pair(2, 0) );  A_vals.push_back( 1 );
  A_indexes.push_back( std::make_pair(2, 1) );  A_vals.push_back( 2 );
  A_indexes.push_back( std::make_pair(2, 2) );  A_vals.push_back( 3 );
  
  std::vector<double> b;
  b.push_back(5); b.push_back(6); b.push_back(14);
  std::vector<double> x;
  x.reserve( b.size() );

  solver.Solve(A_indexes, A_vals, b, &x);
  EXPECT_DOUBLE_EQ( 1, x[0] );
  EXPECT_DOUBLE_EQ( 2, x[1] );
  EXPECT_DOUBLE_EQ( 3, x[2] );
}