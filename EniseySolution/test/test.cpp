// EXPECT(expected,actual) - �� ��������� ������, ��������������� ������������.
// ASSERT - ��������� ������, ������������, ����� ����������� �� ����� ������.
#include "gtest/gtest.h"

TEST(test_case_name, test_name) {
  EXPECT_EQ(1, 2) << "Nonsense!";
  EXPECT_EQ(2, 3) << "Another Nosense!";
  ASSERT_EQ(3, 4) << "I can't take this!";
  EXPECT_EQ(4, 5) << "This is impossible!";
}

 int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}





// �� London
/*
// Testing and mocking
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <functional>
#include <list>

// ��� getch()
#include "conio.h"

// London
#include "gas.h"
#include "gas_count_functions.cuh"

#include "passport_pipe.h"
#include "manager_edge_model_pipe_sequential.h"
#include "model_pipe_sequential.h"
#include "loader_sardan.h"
#include "model_pipe_sequential_functions_proto.cuh"
#include "manager_edge_model_pipe_sequential_cuda.cuh"

#include "utils.h"

TEST(Gas, GasCountFunctions)
{
	// ��������� ������������ ������ ������� �������.
	// ���� ����� - � ����� ����������� ������������� ���-2006
	// � ��������� ������ ������� ����������� ������ ��������.
	// ���� ���������� �������� ������, ��������� ������������ ���� ��������
	// �� ������ - � ��������� ��� ����������� ��������� � ����.
	// � ����� ������������ �������� ��� z, c, mju.

	// ����� �������� ����, ��� �������� �������� ������������.
	GasCompositionReduced composition;
	GasWorkParameters params;
	composition.density_std_cond = 0.6865365; // [��/�3]
	composition.co2 = 0;
	composition.n2 = 0;
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
	
	// �������� �� ������������ ��� �� �������, ������� ������� � �� ��������� �����
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
	ASSERT_LE(abs(t_pseudo_critical - 194.07213), eps);
	ASSERT_LE(abs(p_pseudo_critical - 4.6355228), eps);
	ASSERT_LE(abs(z_standart_conditions - 0.99798650), eps);
	ASSERT_LE(abs(r_standart_conditions - 503.37848), eps);

	// �������� ���������� ��� ������� ��������
	ASSERT_LE(abs(p_reduced - 0.64219296), eps);
	ASSERT_LE(abs(t_reduced - 1.4416289), eps);
	ASSERT_LE(abs(c - 2372.5037), eps);
	ASSERT_LE(abs(di - 5.0115082e-6), eps);
	ASSERT_LE(abs(mju - 1.0930206e-5), eps);
	ASSERT_LE(abs(z - 0.91878355), eps);
	ASSERT_LE(abs(ro - 23.002302), eps);

	// �������� ��� re � lambda
	// ��� re ����� ���� ��������, ������ ��� ����� �������
	float eps_re = 1;
	ASSERT_LE(abs(re - 31017164.0), eps_re);
	ASSERT_LE(abs(lambda - 0.010797811), eps);
}

TEST(PipeSequential, CountSequentialOut)
{
	// ���� ������� ����� ������� ����������������� �����
	// ����������� ����� ��� - ������� ������� ������, ������� ��������� ���-��
	// � ������� �� � ���-�� ����� ��� ����� �� ������� ������.
	// ���� ���������� �������� ��������, �������� �� � � ���������� �����
	// ������������ � �������� �������.

	// ����� ����������� �������� �����.
 	PassportPipe passport;
	FillTestPassportPipe(&passport);

	// ����� �������� ���� �� �����.
	GasCompositionReduced composition;
	GasWorkParameters params_in;
	composition.density_std_cond = 0.6865365; // [��/�3]
	composition.co2 = 0;
	composition.n2 = 0;
	params_in.p = 5; // [���]
	params_in.t = 293.15; // [�]
	params_in.q = 387.843655734; // [�3/���]

	// ���������� ������ ���������� ���� �� ������.
	float p_out;
	float t_out;

	float number_of_segments = 10;
	float length_of_segment = passport.length_ / number_of_segments;
	// �������� ���������� p_out = 2.9721224, t_out = 280.14999 (� ����� - p_out = 2.9769, t_out = 279.78)
	// ���������� ����� ������ �� �����, ��������� �� �� ������.
	// ����� ����������� - � ���� ����������� ������� �� ���, � � ����� - ���������� ������.
	// ToDo: ����������� ����� � ���� ������������.
	FindSequentialOut(
		params_in.p, params_in.t, params_in.q,  // ������� ��������� �������� ������ �� �����
		composition.density_std_cond, composition.co2, composition.n2, // ������ ����
		passport.d_inner_, passport.d_outer_, passport.roughness_coeff_, passport.hydraulic_efficiency_coeff_, // ��-�� �����
		passport.t_env_, passport.heat_exchange_coeff_, // ��-�� ������� ����� (���� ������ � �������� �����)
		length_of_segment, number_of_segments, // ����� �������� � ���-�� ���������
		&t_out, &p_out); 

	float eps = 1.0e-4;
	ASSERT_LE(abs(p_out - 2.9721224), eps);
	ASSERT_LE(abs(t_out - 280.14999), eps);
}

TEST(PipeSequential, Count)
{
	// ��������� ������ �����.
	// �������� ������������ ������� - �������� � ������, ���� ������ - 
	// ��������� ���������� ��������� � ������� �� ������.
	// ����� ����������� �������� �����.
 	PassportPipe passport;
	FillTestPassportPipe(&passport);

	// ����� �������� ���� �� �����.
	Gas gas_in;
	FillTestGasIn(&gas_in);

	// ����� ��������� ���� �� ������
	Gas gas_out;
	FillTestGasOut(&gas_out);

	// ������ ������ �����
	ModelPipeSequential pipe(&passport);
	pipe.set_gas_in(&gas_in);
	pipe.set_gas_out(&gas_out);
	pipe.Count();

	// ��������� ����� ��� ������ ������������ - q = 387.84
	// ��������� ����� ������� - 385.8383, ��� ������ �� �����, �� �� ���������.
	// ��������� ��� ������� � �������� ���������� - ��� ���������� ����������, ����
	// ����� ���������������, ��� ���-�� ����������, ����� �������� � �����������.
	float eps = 0.1;
	ASSERT_LE(abs(pipe.q() - 385.83), eps);
}

TEST(Cuda, Cuda)
{
	PassportPipe passport;
	FillTestPassportPipe(&passport);
	Gas gas_in;
	FillTestGasIn(&gas_in);
	Gas gas_out;
	FillTestGasOut(&gas_out);

	ManagerEdgeModelPipeSequentialCuda manager_cuda;

	std::vector<Edge*> edges;
	edges.resize(4);

	for(int i = 0; i < 4; i++)
	{
		edges[i] = manager_cuda.CreateEdge(&passport);
	}

	std::for_each(edges.begin(), edges.end(), [gas_in, gas_out](Edge* edge)
	{
		edge->set_gas_in(&gas_in);
		edge->set_gas_out(&gas_out);
	});

	// ������������ �������. ����� �� ���������� ������, ���������?
	//edges[0]->set_gas_in(&gas_in);
	//edges[0]->set_gas_out(&gas_out);

	//edges[1]->set_gas_in(&gas_in);
	//edges[1]->set_gas_out(&gas_out);

	manager_cuda.CountAll();
}


//TEST(ManagerEdgeModelPipeSequential, LoadTest)
//{
//	// �������� � ��������� 50 000 ����, � ���������� �� 100 ���
//	// ���������� ����� ��� �������
//	PassportPipe &passport;
//	FillTestPassportPipe(PassportPipe* passport)

//	// ����� �������� ���� �� �����.
//	GasCompositionReduced composition;
//	GasWorkParameters params_in;
//	composition.density_std_cond = 0.6865365; // [��/�3]
//	composition.co2 = 0;
//	composition.n2 = 0;
//	params_in.p = 5; // [���]
//	params_in.t = 293.15; // [�]
//	params_in.q = 387.843655734; // [�3/���]
//	Gas gas_in;
//	gas_in.composition = composition;
//	gas_in.work_parameters = params_in;

//	// ����� ��������� ���� �� ������
//	Gas gas_out = gas_in;
//	gas_out.work_parameters.p = 3;
//	gas_out.work_parameters.t = 0;
//	gas_out.work_parameters.q = 0;

//	Edge *edge;
//	ManagerEdgeModelPipeSequential manager;

//	// ��������� �������� ���������
//	for(int i = 50000; i > 0; --i)
//	{
//		edge = manager.CreateEdge(&passport);
//		edge->set_gas_in(&gas_in);
//		edge->set_gas_out(&gas_out);
//	}

//	// ��������� ������ 100 ��������
//	std::cout << "Performing 10 iterations..." << std::endl;
//	for(int i = 1; i <= 10; ++i)
//	{
//		manager.CountAll();
//		std::cout << i << " iterations performed" << std::endl;
//	}
//}


//TEST(LoaderSardan, LoaderSardan)
//{
//	loader LoaderSardan;
//	loader.Load();
//}

int main(int argc, char** argv) {
  // The following line must be executed to initialize Google Mock
  // (and Google Test) before running the tests.
  ::testing::InitGoogleMock(&argc, argv);
  
  int result = RUN_ALL_TESTS();

  // ������� ������� ������� � ���������� ���������, � ������� 
  // ����������� �����
  std::cout << "Press any key..." << std::endl;
  _getch();
  return result;
}
����� London
*/ 