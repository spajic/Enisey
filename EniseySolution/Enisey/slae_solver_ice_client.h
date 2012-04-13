/** \file slae_solver_ice_client.h
  �������� ����, ������������ � ������� ICE.*/
#pragma once
#include "slae_solver_i.h"
#include <vector>
class SlaeSolverIceClient : public SlaeSolverI {
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