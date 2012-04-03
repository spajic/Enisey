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
  /** ���������� ������ ������� q � t��� �� (p_��, t_��, ������_����_��, p_���)
  �� ���� ��� ����, ����� ����� ���� ��������� Count ����� �������������� 
  ��������� set_gas_in, set_gas_out, ������ ����� ����� ����������� ���������
  �� ����� � ������. 
  ���� p_�� < p_���, �� ����� ��������� �����������, ��� ����� � ��������
  �������. � ���� ������ gas_out ��������� ������, � gas_in - �������, ������
  ������������ �������������.*/
  void Count();
  /// �������� ����������� ������.
  float q();
 private:
  bool IsReverse(); ///< ����������� ������ ��������? (�.�. ������ � ��������)
  float q_; ///< ������.
  PassportPipe passport_; ///< ������� �����.
  Gas gas_in_; ///< ������ ��� �� �����.
  Gas gas_out_; ///< ������ ��� �� ������.
};