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
  ������������ �������������.
  ��� �� ��� ������� ��������� ������ �����������: dq_dp_in, dq_dp_out.
  dq_dp = ( q(p + eps) - q(p) ) / eps.
  */
  void Count();
  float q(); ///< �������� ����������� ������.
  float dq_dp_in(); ///< �������� ����������� �� P��.
  float dq_dp_out(); ///< �������� ����������� �� P���.
 private:
  bool IsReverse(); ///< ����������� ������ ��������? (�.�. ������ � ��������)
  /// ����� ������� ������� q � ����������������� �����������.
  void CallFindSequentialQ(
      const Gas &gas_in,
      const Gas &gas_out,
      const PassportPipe &passport,
      const int number_of_segments,
      float *t_out,
      float *q_out); 
  float q_; ///< ������.
  float dq_dp_in_; ///< ����������� q �� P��.
  float dq_dp_out_; ///< ����������� q �� P���.
  PassportPipe passport_; ///< ������� �����.
  Gas gas_in_; ///< ������ ��� �� �����.
  Gas gas_out_; ///< ������ ��� �� ������.
};