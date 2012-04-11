/** \file test_slae_solver_cvm.cpp
  �������� ���� �� ���� CVM.*/
#pragma once
#include <vector>
class SlaeSolverCVM {
 public:
  /** ������ ���� AX = B. 
  �������������� ����������� ����, ��� ����� ��������� ������ ������� 
  ��������� ������������� A, � ��������������� ������� A_values - ��
  ��������.*/
  void Solve(
      std::vector<std::pair<int, int> > const &A_indexes,
      std::vector<double> const &A_values,
      std::vector<double> const &B,
      std::vector<double> *X);
};