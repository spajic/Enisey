/** \file model_pipe_sequential.h
����� ModelPipeSequential - ������ ����������������� ����� ��� �����.*/
#pragma once
#include "gas.h"
#include "passport_pipe.h"

/** ������ ����������������� ����� ��� �����. 
������������ � ManagerEdgeModelPipeSequential � �������� "������� �������". */
class ModelPipeSequential {
 public:
  /// ����������� �� ���������. ������ �� ������.
  ModelPipeSequential();
  /// �������� ���������� ������ passport � ���� PassportPipe � ���������.
  /// \todo ����� ����� ����� ����� ����� PassportPipe.
  ModelPipeSequential(const Passport* passport);
  /// ���������� ������ ��� �� ����� �����.
  void set_gas_in(const Gas* gas);
  /// ���������� ������ ��� �� ������ �����.
  void set_gas_out(const Gas* gas);
  /// ���������� ������ ������� q � t��� �� (p_��, t_��, ������_����_��, p_���)
  void Count();
  /// �������� ����������� ������.
  float q();
 private:
  void DetermineDirectionOfFlow(); ///< ���������� ����������� ������.
  bool direction_is_forward_; ///< ����������� ������ - ������ ��� ��������.
  float q_; ///< ������.
  PassportPipe passport_; ///< ������� �����.
  Gas gas_in_; ///< ������ ��� �� �����.
  Gas gas_out_; ///< ������ ��� �� ������.
};