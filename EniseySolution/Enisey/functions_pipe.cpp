#include "functions_pipe.h"
#include "math.h"

#pragma once
/*! \file functions_pipe.h
 *  \brief ������� ������� �����.
 *
 *  ����� ��������� �������� ����� ����� ���� ��������� �����.
 */
/// \todo ���������������� � LaTex-���������, �������� � �������.
/// \todo ������� � ���� ������������ ���������.
#include "math.h"
#include "functions_gas.h"
#include <iostream>
#include <boost/lexical_cast.hpp>


double ReturnPNextSequential(
    double p_work, 
    double t_work, 
    double q, // ������� ��������� �������� ������
    double den_sc, // ������ ����
    double r_sc,   // ������� �������� ����          			
    double lambda, 
    double z, // �������� ���� ��� ������� ��������
    double d_inner, // �������� �����
    double length_of_segment) { // ����� ��������
  // static const double kPi = 3.14159265358979323846264338327950288419716939937510;
  // ���������� ��� ������� p_next
  double minus = 10*q*q * (length_of_segment) * 
    (16*r_sc * (den_sc*den_sc) * lambda * z * t_work) / 
    (3.1415926535897932384626433832795*3.1415926535897932384626433832795 *
    pow(d_inner/10, 5) );
  double p_next = p_work*p_work - minus;
  // ���� ���������� ������� �������� ������ ����,
  // ���������� ���������� ������ �� ����
  if(p_next > 0) {
    return pow(p_next, static_cast<double>(0.5));		 
  }
  // ����� - �������, ��� �������� ����� �� ����.
  else {
    p_next = 0;
  }
  return p_next;
};


double ReturnTNextSequential(
    double p_next, // ��������� ������� ReturnPNextSequential
    double p_work, 
    double t_work, 
    double q, // ������� ��������� �������� ������
    double den_sc, // ������ ����
    double c, 
    double di, // �������� ���� ��� ������� ��������
    double d_outer, // �������� �����
    double t_env, 
    double heat_exchange_coeff, // �������� ����������� �����
    double length_of_segment) {
  // kPi = 3.14159265358979323846264338327950288419716939937510
  // ����-� ��� (T-Tos) [�.�]
  double s1 = (length_of_segment * d_outer * heat_exchange_coeff * 3.1415926535897932384626433832795) / 
    (c * q * den_sc);
  // ����-� ��� (PNext - P) [�/���] 
  double s2 = di*1000000; 
  double t_next = t_work - s1 * (t_work - t_env) + s2 * (p_next - p_work);
  // ���� ���������� ����������� ������ ����������� ���������� �����, �� 
  // �������, ��� ����� �� ����������� ���������� �����
  if(t_next < t_env) {
    t_next = t_env;
  }
  return t_next;
};

// ���������� ��������� �������� ������ �� ������ �����
// �� ��������� ����, ������� ���������� �� ����� �����, ��������� �����, ��-��� ������� �����, ���������� ���������

void FindSequentialOut(
    double p_work, 
    double t_work, 
    double q,  // ������� ��������� �������� ������ �� �����
    double den_sc, 
    double co2, 
    double n2, // ������ ����
    double d_inner, 
    double d_outer, 
    double roughness_coeff, 
    double hydraulic_efficiency_coeff, // ��-�� �����
    double t_env, 
    double heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
    double length_of_segment, 
    int number_of_segments, // ����� �������� � ���-�� ���������
    double* t_out,
    double* p_out) { // out - ���������, �������� �� ������ 
  // � ����� ��������������� ������������ ����� � "������� ������"
  double p_next = p_work;
  double t_next = t_work;
  double p_current = p_work;
  double t_current = t_work;

  double p_pseudo_critical = FindPPseudoCritical(den_sc, co2, n2);
  double t_pseudo_critical = FindTPseudoCritical(den_sc, co2, n2);
  double r_sc = FindRStandartConditions(den_sc);

  double p_reduced = 0;
  double t_reduced = 0;
  double z = 0;
  double c = 0;
  double mju = 0;
  double di = 0;
  double re = 0;
  double lambda = 0;

  for(int i = number_of_segments; i != 0; --i) {
    p_current = p_next;
    t_current = t_next;
    // ��������� ����������� �������� ���� ��� ������� ������� ��������
    p_reduced = FindPReduced(p_current, p_pseudo_critical);
    t_reduced = FindTReduced(t_current, t_pseudo_critical);

    z = FindZ(p_reduced, t_reduced);
    c = FindC(t_reduced, p_reduced, r_sc);
    di = FindDi(p_reduced, t_reduced);
    mju = FindMju(p_reduced, t_reduced);

    re = FindRe(q, den_sc, mju, d_inner);
    lambda = FindLambda(re, d_inner, roughness_coeff, hydraulic_efficiency_coeff);
    // ��������� �������� P � T � ��������� ����
    p_next = ReturnPNextSequential(p_current, t_current, q, den_sc, r_sc, lambda, z, d_inner, length_of_segment);
    t_next = ReturnTNextSequential(p_next, p_current, t_current, q, den_sc, c, di, d_outer, t_env, heat_exchange_coeff, length_of_segment);
  }

  // ����������� ������������ �������� � out-���������	
  *p_out = p_next;
  *t_out = t_next;
};

// ����� ����� q, ����� ������ ��������� P���(P��, T��, q) = p_target ������������ q.
// ������� P��� � ��� ���� - FindSequentialOut.
// �������� �������, EquationToSolve(q) = p_target - P���(P���, T��, q).
// ��� ����� ����� ����� q0, ��� EquationToSolve(q0) = 0.
// �����, ����� ������� �� q ����������
// � ����� ������� ������� ���� ������������, ��������� ����� ����,
// � � ����� ������� ���� ������������. (��� ������� ������� ������� ������� �������).

double EquationToSolve(
    double p_target,
    double p_work, 
    double t_work, 
    double q,  // ������� ��������� �������� ������ �� �����
    double den_sc, 
    double co2, 
    double n2, // ������ ����
    double d_inner, 
    double d_outer, 
    double roughness_coeff, 
    double hydraulic_efficiency_coeff, // ��-�� �����
    double t_env, 
    double heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
    double length_of_segment, 
    int number_of_segments, // ����� �������� � ���-�� ���������
    double* t_out, 
    double* p_out) {
  FindSequentialOut(
      p_work, 
      t_work, 
      q,  // ������� ��������� �������� ������ �� �����
      den_sc, 
      co2, 
      n2, // ������ ����
      d_inner, 
      d_outer, 
      roughness_coeff, 
      hydraulic_efficiency_coeff, // ��-�� �����
      t_env, 
      heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
      length_of_segment, 
      number_of_segments, // ����� �������� � ���-�� ���������
      t_out, 
      p_out); // out - ���������, �������� �� ������ 
  return p_target - *p_out;
}

// ��������� ��� ���� ������� - p_target + ��� �� �, ��� ��� FindSequentialOut
// �� ��������� ����������� - q - ������ out-��������, ������� ������� p_out

int FindSequentialQ(
  double p_target, // ��������, ������� ������ ���������� � �����
  double p_work, double t_work,  // ������� ��������� �������� ������ �� �����
  double den_sc, double co2, double n2, // ������ ����
  double d_inner, double d_outer, double roughness_coeff, double hydraulic_efficiency_coeff, // ��-�� �����
  double t_env, double heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
  double length_of_segment, int number_of_segments, // ����� �������� � ���-�� ���������
  double* t_out, double* q_out) {// out - ���������, �������� �� ������ )
  // ������ ��������� ������� ������� ������� �������
  // ��������� ������ - start, finish - ���������� �������, ��� ������ �������
  // eps_x, eps_y - �������� ��� ������� ������
  // ToDo: ������� ��������� ������ ����������� �������.
  // ��������, ��� ������� ��������� ����� ������� ������� ������� ��� ������� 
  // ������ ������.
  // ToDo: ��������� ������������ ������������ �������� (��� ��������� �������).
  double start = 0.0000000001;
  double finish = 100000;
  double eps_x = 0.0000001;
  double eps_y = 0.000001;

  // �������� ��� ������ ������� EquationToSolve
  double p_out;
  // ��������
  // ��������������, ��� ������� ������ ����������, ���������� ���� ����, ������������� ���� ����.
  if(EquationToSolve(
    p_target,
    p_work, t_work, start,  // ������� ��������� �������� ������ �� �����
    den_sc, co2, n2, // ������ ����
    d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // ��-�� �����
    t_env, heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
    length_of_segment, number_of_segments, // ����� �������� � ���-�� ���������
    t_out, &p_out) > 0 )
  {
    //throw "Error. f(start) must be negative";
    return -1;
  }
  if(EquationToSolve(
    p_target,
    p_work, t_work, finish,  // ������� ��������� �������� ������ �� �����
    den_sc, co2, n2, // ������ ����
    d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // ��-�� �����
    t_env, heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
    length_of_segment, number_of_segments, // ����� �������� � ���-�� ���������
    t_out, &p_out) < 0)
  {
    //throw "Error. f(finish) must be positive";
    return -2;
  }
  if(start > finish) {
    //throw "Error. Start must be less when finish";
    return -3;
  }

  // local values
  double a = start;
  double b = finish;

  double middle = (a + b) / 2;
  double middle_val = EquationToSolve(
    p_target,
    p_work, t_work, middle,  // ������� ��������� �������� ������ �� �����
    den_sc, co2, n2, // ������ ����
    d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // ��-�� �����
    t_env, heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
    length_of_segment, number_of_segments, // ����� �������� � ���-�� ���������
    t_out, &p_out);
  int iters = 0;
  while(/*abs( middle_val ) > eps_y && abs(a - b) > eps_x &&*/ iters < 40) {
    iters++;
    if(middle_val < 0) 
      a = middle;
    else
      b = middle;

    middle = (a + b) / 2;
    middle_val = EquationToSolve(
      p_target,
      p_work, t_work, middle,  // ������� ��������� �������� ������ �� �����
      den_sc, co2, n2, // ������ ����
      d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // ��-�� �����
      t_env, heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
      length_of_segment, number_of_segments, // ����� �������� � ���-�� ���������
      t_out, &p_out);
  }

  // ���������� ��������� � out-��������.
  *q_out = middle; 
  //std::cout << ("q= " +boost::lexical_cast<std::string>(middle) + " iters " + 
  //  boost::lexical_cast<std::string>(iters) + "\n");

  // ���������� ��� ���������� ��� ������.
  return 0;
};