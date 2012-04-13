/** \file test_slae_solver_cvm.cpp
  ����� ��� ������ SlaeSolverCVM �� slae_solver_cvm.h*/
#include "slae_solver_cvm.h"

#include "gtest/gtest.h"
#include "test_utils.h"

#include <vector>
#include "test_utils.h"

TEST(SlaeSolverCVMTest, SolvesSimpleSlae) {
/* ��������� �������� ������� �� ���� CVM.cpp
������ ����
[2 0 1]   [1]   [5]
[0 3 0] * [2] = [6]
[1 2 3]   [3]   [14]
������ �������� ����������:
  A_indexes - ������� (������, �������) ��������� ����-�� �.
  A_vals - �������� ���� ����-�� � ��������������� �������.
  B - ������ b.
  X - ������ ���������� �������. 
��� ������� ��� ������������� �������� ����������� ������. 
���� ��� ���, �������, ���� ���� ������� ���� CSR.*/
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
