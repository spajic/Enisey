/** \file test_slae_solver_cvm.h
  �������� ���� �� ���� CVM.*/
#pragma once
#include "slae_solver_i.h"
#include <vector>
class SlaeSolverCVM : public SlaeSolverI {
 public:
  /** ������ ���� AX = B. 
  �������������� ����������� ����, ��� ����� ��������� ������ ������� 
  ��������� ������������� A, � ��������������� ������� A_values - ��
  ��������.*/
  virtual void Solve(
      std::vector<int> const &A_indexes,
      std::vector<double> const &A_values,
      std::vector<double> const &B,
      std::vector<double> *X);
};