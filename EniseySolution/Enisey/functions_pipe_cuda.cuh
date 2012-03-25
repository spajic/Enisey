#pragma once

#pragma once

#include "math.h"

#include "gas_count_functions_cuda.cuh"

__device__
inline float ReturnPNextSequentialCuda(
	float p_work, float t_work, float q, // ������� ��������� �������� ������
	float den_sc, // ������ ����
	float r_sc,   // ������� �������� ����          			
	float lambda, float z, // �������� ���� ��� ������� ��������
	float d_inner, // �������� �����
	float length_of_segment) // ����� ��������
{
	// static const float kPi = 3.14159265358979323846264338327950288419716939937510;
	// ���������� ��� ������� p_next
	float minus = 10*q*q * (length_of_segment) * 
		(16*r_sc * (den_sc*den_sc) * lambda * z * t_work) / 
		(3.1415926535897932384626433832795*3.1415926535897932384626433832795 *
		pow(d_inner/10, 5) );
	float p_next = p_work*p_work - minus;
	// ���� ���������� ������� �������� ������ ����,
	// ���������� ���������� ������ �� ����
	if(p_next > 0)
	{
		return pow(p_next, static_cast<float>(0.5));		 
	}
	// ����� - �������, ��� �������� ����� �� ����.
	else
	{
		p_next = 0;
	}
	return p_next;
};

__device__
inline float ReturnTNextSequentialCuda(
	float p_next, // ��������� ������� ReturnPNextSequential
	float p_work, float t_work, float q, // ������� ��������� �������� ������
	float den_sc, // ������ ����
	float c, float di, // �������� ���� ��� ������� ��������
	float d_outer, // �������� �����
	float t_env, float heat_exchange_coeff, // �������� ����������� �����
	float length_of_segment)
{
	// kPi = 3.14159265358979323846264338327950288419716939937510
	// ����-� ��� (T-Tos) [�.�]
	float s1 = (length_of_segment * d_outer * heat_exchange_coeff * 3.1415926535897932384626433832795) / 
		(c * q * den_sc);

	// ����-� ��� (PNext - P) [�/���] 
	float s2 = di*1000000; 
	float t_next = t_work - s1 * (t_work - t_env) + s2 * (p_next - p_work);
	// ���� ���������� ����������� ������ ����������� ���������� �����, �� 
	// �������, ��� ����� �� ����������� ���������� �����
	if(t_next < t_env)
	{
		t_next = t_env;
	}
	return t_next;
};


__device__
inline void FindSequentialOutCudaRefactored(
	float p_work, float t_work, float q,  // ������� ��������� �������� ������ �� �����
	float p_pc, float t_pc, float r_sc, float den_sc,
	float d_inner, float d_outer, float roughness_coeff, float hydraulic_efficiency_coeff, // ��-�� �����
	float t_env, float heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
	float length_of_segment, int number_of_segments, // ����� �������� � ���-�� ���������
	float* p_out, float* t_out) // out - ���������, �������� �� ������ 
{
	// � ����� ��������������� ������������ ����� � "������� ������"
	float p_next = p_work;
	float t_next = t_work;
	float p_current = p_work;
	float t_current = t_work;

	float p_reduced = 0;
	float t_reduced = 0;
	float z = 0;
	float c = 0;
	float mju = 0;
	float di = 0;
	float re = 0;
	float lambda = 0;
		
	for(int i = number_of_segments; i != 0; --i)
	{
		p_current = p_next;
		t_current = t_next;

		// ��������� ����������� �������� ���� ��� ������� ������� ��������
		p_reduced = FindPReducedCuda(p_current, p_pc);
		t_reduced = FindTReducedCuda(t_current, t_pc);
		
		z = FindZCuda(p_reduced, t_reduced);
		c = FindCCuda(t_reduced, p_reduced, r_sc);
		di = FindDiCuda(p_reduced, t_reduced);
		mju = FindMjuCuda(p_reduced, t_reduced);

		re = FindReCuda(q, den_sc, mju, d_inner);
		lambda = FindLambdaCuda(re, d_inner, roughness_coeff, hydraulic_efficiency_coeff);

		// ��������� �������� P � T � ��������� ����
		p_next = ReturnPNextSequentialCuda(p_current, t_current, q, den_sc, r_sc, lambda, z, d_inner, length_of_segment);
		t_next = ReturnTNextSequentialCuda(p_next, p_current, t_current, q, den_sc, c, di, d_outer, t_env, heat_exchange_coeff, length_of_segment);
	}

	// ����������� ������������ �������� � out-���������	
	*p_out = p_next;
	*t_out = t_next;
};

__device__
inline float EquationToSolveCudaRefactored(
	float p_target,
	float p_work, float t_work, float q,  // ������� ��������� �������� ������ �� �����
	float p_pc, float t_pc, float r_sc, float den_sc,
	float d_inner, float d_outer, float roughness_coeff, float hydraulic_efficiency_coeff, // ��-�� �����
	float t_env, float heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
	float length_of_segment, int number_of_segments, // ����� �������� � ���-�� ���������
	float* p_out, float* t_out) // out - ���������, �������� �� ������ 
{
	FindSequentialOutCudaRefactored(
		 p_work,  t_work,  q,  // ������� ��������� �������� ������ �� �����
		 p_pc,  t_pc,  r_sc,  den_sc,
		 d_inner,  d_outer,  roughness_coeff,  hydraulic_efficiency_coeff, // ��-�� �����
		 t_env,  heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
		 length_of_segment, number_of_segments, // ����� �������� � ���-�� ���������
		 p_out, t_out); // out - ���������, �������� �� ������ 
	return p_target - *p_out;
}

__device__
inline float FindSequentialQCudaRefactored(
	float p_target,
	float p_work, float t_work,  // ������� ��������� �������� ������ �� �����
	float p_pc, float t_pc, float r_sc, float den_sc,
	float d_inner, float d_outer, float roughness_coeff, float hydraulic_efficiency_coeff, // ��-�� �����
	float t_env, float heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
	float length_of_segment, int number_of_segments, // ����� �������� � ���-�� ���������
	float* p_out, float* t_out,
	float* q_out) // out - ���������, �������� �� ������ 
{
	// ������ ��������� ������� ������� ������� �������
	// ��������� ������ - start, finish - ���������� �������, ��� ������ �������
	// eps_x, eps_y - �������� ��� ������� ������
	// ToDo: ������� ��������� ������ ����������� �������.
	// ��������, ��� ������� ��������� ����� ������� ������� ������� ��� ������� 
	// ������ ������.
	// ToDo: ��������� ������������ ������������ �������� (��� ��������� �������).
	float a = 200; // start
	float b = 400; // finish
	float eps_x = 0.1;
	float eps_y = 0.001;

	// �������� ��� ������ ������� EquationToSolve
	//float p_out;
	// ��������
	// ��������������, ��� ������� ������ ����������, ���������� ���� ����, ������������� ���� ����.
	/*if(EquationToSolveCudaRefactored(
		p_target,
		p_work, t_work, a,  // ������� ��������� �������� ������ �� �����
		p_pc, t_pc, r_sc, den_sc,
		d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // ��-�� �����
		t_env, heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
		length_of_segment, number_of_segments, // ����� �������� � ���-�� ���������
		t_out, &p_out) > 0 )
	{
		//throw "Error. f(start) must be negative";
		return -1;
	}
	if(EquationToSolveCudaRefactored(
		p_target,
		p_work, t_work, b,  // ������� ��������� �������� ������ �� �����
		p_pc, t_pc, r_sc, den_sc,
		d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // ��-�� �����
		t_env, heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
		length_of_segment, number_of_segments, // ����� �������� � ���-�� ���������
		t_out, p_out) < 0)
	{
		//throw "Error. f(finish) must be positive";
		return -2;
	}
	if(start > finish)
	{
		//throw "Error. Start must be less when finish";
		return -3;
	}*/

	float middle = (a + b) / 2;
	float middle_val = EquationToSolveCudaRefactored(
		p_target,
		p_work, t_work, middle,  // ������� ��������� �������� ������ �� �����
		p_pc, t_pc, r_sc, den_sc,
		d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // ��-�� �����
		t_env, heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
		length_of_segment, number_of_segments, // ����� �������� � ���-�� ���������
		t_out, p_out);

	while(abs( middle_val ) > eps_y && abs(a - b) > eps_x)
	{
		if(middle_val < 0) 
			a = middle;
		else
			b = middle;

		middle = (a + b) / 2;
		middle_val = EquationToSolveCudaRefactored(
			p_target,
			p_work, t_work, middle,  // ������� ��������� �������� ������ �� �����
			p_pc, t_pc, r_sc, den_sc,
			d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // ��-�� �����
			t_env, heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
			length_of_segment, number_of_segments, // ����� �������� � ���-�� ���������
			t_out, p_out);
	}
	
	// ���������� ��������� � out-��������.
	*q_out = middle; 

	// ���������� ��� ���������� ��� ������.
	return 0;
}

// ���������� ��������� �������� ������ �� ������ �����
// �� ��������� ����, ������� ���������� �� ����� �����, ��������� �����, ��-��� ������� �����, ���������� ���������
__device__
inline void FindSequentialOutCuda(
	float p_work, float t_work, float q,  // ������� ��������� �������� ������ �� �����
	float den_sc, float co2, float n2, // ������ ����
	float d_inner, float d_outer, float roughness_coeff, float hydraulic_efficiency_coeff, // ��-�� �����
	float t_env, float heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
	float length_of_segment, int number_of_segments, // ����� �������� � ���-�� ���������
	float* t_out, float* p_out) // out - ���������, �������� �� ������ 
{
	// � ����� ��������������� ������������ ����� � "������� ������"
	float p_next = p_work;
	float t_next = t_work;
	float p_current = p_work;
	float t_current = t_work;

	float p_pseudo_critical = FindPPseudoCriticalCuda(den_sc, co2, n2);
	float t_pseudo_critical = FindTPseudoCriticalCuda(den_sc, co2, n2);
	float r_sc = FindRStandartConditionsCuda(den_sc);

	float p_reduced = 0;
	float t_reduced = 0;
	float z = 0;
	float c = 0;
	float mju = 0;
	float di = 0;
	float re = 0;
	float lambda = 0;
		
	for(int i = number_of_segments; i != 0; --i)
	{
		p_current = p_next;
		t_current = t_next;

		// ��������� ����������� �������� ���� ��� ������� ������� ��������
		p_reduced = FindPReducedCuda(p_current, p_pseudo_critical);
		t_reduced = FindTReducedCuda(t_current, t_pseudo_critical);
		
		z = FindZCuda(p_reduced, t_reduced);
		c = FindCCuda(t_reduced, p_reduced, r_sc);
		di = FindDiCuda(p_reduced, t_reduced);
		mju = FindMjuCuda(p_reduced, t_reduced);

		re = FindReCuda(q, den_sc, mju, d_inner);
		lambda = FindLambdaCuda(re, d_inner, roughness_coeff, hydraulic_efficiency_coeff);

		// ��������� �������� P � T � ��������� ����
		p_next = ReturnPNextSequentialCuda(p_current, t_current, q, den_sc, r_sc, lambda, z, d_inner, length_of_segment);
		t_next = ReturnTNextSequentialCuda(p_next, p_current, t_current, q, den_sc, c, di, d_outer, t_env, heat_exchange_coeff, length_of_segment);
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
__device__
inline float EquationToSolveCuda(
	float p_target,
	float p_work, float t_work, float q,  // ������� ��������� �������� ������ �� �����
	float den_sc, float co2, float n2, // ������ ����
	float d_inner, float d_outer, float roughness_coeff, float hydraulic_efficiency_coeff, // ��-�� �����
	float t_env, float heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
	float length_of_segment, int number_of_segments, // ����� �������� � ���-�� ���������
	float* t_out, float* p_out)
{
	FindSequentialOutCuda(
		p_work, t_work, q,  // ������� ��������� �������� ������ �� �����
		den_sc, co2, n2, // ������ ����
		d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // ��-�� �����
		t_env, heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
		length_of_segment, number_of_segments, // ����� �������� � ���-�� ���������
		t_out, p_out); // out - ���������, �������� �� ������ 
	return p_target - *p_out;
}

// ��������� ��� ���� ������� - p_target + ��� �� �, ��� ��� FindSequentialOut
// �� ��������� ����������� - q - ������ out-��������, ������� ������� p_out
__device__
inline int FindSequentialQCuda(
	float p_target, // ��������, ������� ������ ���������� � �����
	float p_work, float t_work,  // ������� ��������� �������� ������ �� �����
	float den_sc, float co2, float n2, // ������ ����
	float d_inner, float d_outer, float roughness_coeff, float hydraulic_efficiency_coeff, // ��-�� �����
	float t_env, float heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
	float length_of_segment, int number_of_segments, // ����� �������� � ���-�� ���������
	float* t_out, float* q_out) // out - ���������, �������� �� ������ )
{
	// ������ ��������� ������� ������� ������� �������
	// ��������� ������ - start, finish - ���������� �������, ��� ������ �������
	// eps_x, eps_y - �������� ��� ������� ������
	// ToDo: ������� ��������� ������ ����������� �������.
	// ��������, ��� ������� ��������� ����� ������� ������� ������� ��� ������� 
	// ������ ������.
	// ToDo: ��������� ������������ ������������ �������� (��� ��������� �������).
	float start = 0.1;
	float finish = 10000;
	float eps_x = 0.1;
	float eps_y = 0.0001;

	// �������� ��� ������ ������� EquationToSolve
	float p_out;
	// ��������
	// ��������������, ��� ������� ������ ����������, ���������� ���� ����, ������������� ���� ����.
	if(EquationToSolveCuda(
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
	if(EquationToSolveCuda(
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
	if(start > finish)
	{
		//throw "Error. Start must be less when finish";
		return -3;
	}

	// local values
	float a = start;
	float b = finish;

	float middle = (a + b) / 2;
	float middle_val = EquationToSolveCuda(
		p_target,
		p_work, t_work, middle,  // ������� ��������� �������� ������ �� �����
		den_sc, co2, n2, // ������ ����
		d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // ��-�� �����
		t_env, heat_exchange_coeff, // ��-�� ������� ����� (���� ������ � �������� �����)
		length_of_segment, number_of_segments, // ����� �������� � ���-�� ���������
		t_out, &p_out);

	while(abs( middle_val ) > eps_y && abs(a - b) > eps_x)
	{
		if(middle_val < 0) 
			a = middle;
		else
			b = middle;

		middle = (a + b) / 2;
		middle_val = EquationToSolveCuda(
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

	// ���������� ��� ���������� ��� ������.
	return 0;
};