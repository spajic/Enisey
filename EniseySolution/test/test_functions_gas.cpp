#include "gtest/gtest.h"
#include "test_utils.h"

#include "model_pipe_sequential.h"
#include "gas.h"
#include "functions_pipe.h"
#include "passport_pipe.h"
#include "edge.h"
#include "manager_edge_model_pipe_sequential.h"
#include "loader_vesta.h"
TEST(Gas, GasCountFunctions) {
  // ��������� ������������ ������ ������� �������.
  // ���� ����� - � ����� ����������� ������������� ���-2006
  // � ��������� ������ ������� ����������� ������ ��������.
  // ���� ���������� �������� ������, ��������� ������������ ���� ��������
  // �� ������ - � ��������� ��� ����������� ��������� � ����.
  // � ����� ������������ �������� ��� z, c, mju.

  // ����� �������� ����, ��� �������� �������� ������������.
  GasCompositionReduced composition;
  composition.density_std_cond = 0.6865365; // [��/�3]
  composition.co2 = 0;
  composition.n2 = 0;
  GasWorkParameters params;
  params.p = 2.9769; // [���]
  params.t = 279.78; // [�]
  params.q = 387.843655734; // [�3/���]

  // ���������� ������� ������������� ���� ������ �� �������
  // t_pseudo_critical = 194.07213 [�]
  // p_pseudo_critical = 4.6355228 [���]
  // z_standart_conditions = 0.99798650 [�.�.]
  // r_standart_conditions = 503.37848 [�� / (�� * �)]
  float t_pseudo_critical = FindTPseudoCritical(
    composition.density_std_cond, composition.co2, composition.n2);
  float p_pseudo_critical = FindPPseudoCritical(
    composition.density_std_cond, composition.co2, composition.n2);
  float z_standart_conditions = FindZStandartConditions(
    composition.density_std_cond, composition.co2, composition.n2);
  float r_standart_conditions = FindRStandartConditions(
    composition.density_std_cond);

  // ���������� ������������� ���� ��� ������� �������� � �����������
  // p_reduced = 0.64219296
  // t_reduced = 1.4416289
  // c = 2372.5037		(�������� � ����� Cp = 2310.1)
  // di = 5.0115082e-6
  // mju = 1.0930206e-5	(�������� � ����� mju = 1.093e-5)
  // z = 0.91878355		(�������� � ����� z = 0.91878)
  // ro = 23.002302
  float p_reduced = FindPReduced(params.p, p_pseudo_critical);
  float t_reduced = FindTReduced(params.t, t_pseudo_critical);
  float c = FindC(t_reduced, p_reduced, r_standart_conditions);
  float di = FindDi(p_reduced, t_reduced);
  float mju = FindMju(p_reduced, t_reduced);
  float z = FindZ(p_reduced, t_reduced);
  float ro = FindRo(composition.density_std_cond, params.p, params.t, z);

  /* �������� �� ������������ ��� �� �������, 
  ������� ������� � �� �������� �����*/
  PassportPipe passport;
  FillTestPassportPipe(&passport);

  // re = 31017164.0
  // lambda = 0.010797811
  float re = FindRe(params.q, composition.density_std_cond, mju, passport.d_inner_);
  float lambda = FindLambda(re, passport.d_inner_, passport.roughness_coeff_, passport.hydraulic_efficiency_coeff_);

  // �����, ��� �������� mju � z ����������� ������� � ������,
  // � ������� ����������. 
  // ��������� ����������� ������� �� ������ � ������ EXPECT'�
  // ���� �������� ����� ������ ���������� �� �������, ���� ���� ��� �����.

  // ����� �������� ��������� eps
  float eps = 1.0e-4;

  // �������� ������� ����������
  EXPECT_LE(abs(t_pseudo_critical - 194.07213), eps);
  EXPECT_LE(abs(p_pseudo_critical - 4.6355228), eps);
  EXPECT_LE(abs(z_standart_conditions - 0.99798650), eps);
  EXPECT_LE(abs(r_standart_conditions - 503.37848), eps);

  // �������� ���������� ��� ������� ��������
  EXPECT_LE(abs(p_reduced - 0.64219296), eps);
  EXPECT_LE(abs(t_reduced - 1.4416289), eps);
  EXPECT_LE(abs(c - 2372.5037), eps);
  EXPECT_LE(abs(di - 5.0115082e-6), eps);
  EXPECT_LE(abs(mju - 1.0930206e-5), eps);
  EXPECT_LE(abs(z - 0.91878355), eps);
  EXPECT_LE(abs(ro - 23.002302), eps);

  // �������� ��� re � lambda
  // ��� re ����� ���� ��������, ������ ��� ����� �������
  float eps_re = 1;
  EXPECT_LE(abs(re - 31017164.0), eps_re);
  EXPECT_LE(abs(lambda - 0.010797811), eps);
}