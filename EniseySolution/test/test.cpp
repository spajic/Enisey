// EXPECT(expected,actual) - �� ��������� ������, ��������������� ������������.
// ASSERT - ��������� ������, ������������, ����� ����������� �� ����� ������.
//TEST(test_case_name, test_name) {
//  EXPECT_EQ(1, 2) << "Nonsense!";
//  ASSERT_EQ(2, 3) << "I can't take this!";
//  EXPECT_EQ(3, 4) << "This is impossible!";
//}
// The fixture for testing class Foo.
//class FooTest : public ::testing::Test {
//protected:
//  FooTest() {} // You can do set-up work for each test here.
//  virtual ~FooTest() { } // You can do clean-up work that doesn't throw exceptions here.
//  // If the constructor and destructor are not enough for setting up
//  // and cleaning up each test, you can define the following methods:
//  virtual void SetUp() { }
//    // Code here will be called immediately after the constructor (right
//    // before each test).
//  virtual void TearDown() { }
//    // Code here will be called immediately after each test (right
//    // before the destructor).

//  // Objects declared here can be used by all tests in the test case for Foo.
//};
//// Tests that the Foo::Bar() method does Abc.
//TEST_F(FooTest, MethodBarDoesAbc) {
//  const string input_filepath = "this/package/testdata/myinputfile.dat";
//  const string output_filepath = "this/package/testdata/myoutputfile.dat";
//  Foo f;
//  EXPECT_EQ(0, f.Bar(input_filepath, output_filepath));
//}

#include "gtest/gtest.h"
#include "test_utils.h"

#include "model_pipe_sequential.h"
#include "gas.h"
#include "functions_pipe.h"
#include "passport_pipe.h"
#include "edge.h"
#include "manager_edge_model_pipe_sequential.h"
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

TEST(ManagerEdgeModelPipeSequential, LoadTest)
{
  // �������� � ��������� 50 000 ����, � ���������� �� 100 ���
  // ���������� ����� ��� �������
  PassportPipe passport;
  FillTestPassportPipe(&passport);

  // ����� �������� ���� �� �����.
  GasCompositionReduced composition;
  composition.density_std_cond = 0.6865365; // [��/�3]
  composition.co2 = 0;
  composition.n2 = 0;
  GasWorkParameters params_in;
  params_in.p = 5; // [���]
  params_in.t = 293.15; // [�]
  params_in.q = 387.843655734; // [�3/���]
  Gas gas_in;
  gas_in.composition = composition;
  gas_in.work_parameters = params_in;

  // ����� ��������� ���� �� ������
  Gas gas_out = gas_in;
  gas_out.work_parameters.p = 3;
  gas_out.work_parameters.t = 0;
  gas_out.work_parameters.q = 0;

  Edge *edge;
  ManagerEdgeModelPipeSequential manager;

  // ��������� �������� ���������
  for(int i = 50000; i > 0; --i)
  {
    edge = manager.CreateEdge(&passport);
    edge->set_gas_in(&gas_in);
    edge->set_gas_out(&gas_out);
  }

  // ��������� ������ 10 ��������
  std::cout << "Performing 10 iterations..." << std::endl;
  for(int i = 1; i <= 10; ++i)
  {
    manager.CountAll();
    std::cout << i << " iterations performed" << std::endl;
  }
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


TEST(ManagerEdgeModelPipeSequential, LoadTest)
{
	// �������� � ��������� 50 000 ����, � ���������� �� 100 ���
	// ���������� ����� ��� �������
	PassportPipe &passport;
	FillTestPassportPipe(PassportPipe* passport)

	// ����� �������� ���� �� �����.
	GasCompositionReduced composition;
	GasWorkParameters params_in;
	composition.density_std_cond = 0.6865365; // [��/�3]
	composition.co2 = 0;
	composition.n2 = 0;
	params_in.p = 5; // [���]
	params_in.t = 293.15; // [�]
	params_in.q = 387.843655734; // [�3/���]
	Gas gas_in;
	gas_in.composition = composition;
	gas_in.work_parameters = params_in;

	// ����� ��������� ���� �� ������
	Gas gas_out = gas_in;
	gas_out.work_parameters.p = 3;
	gas_out.work_parameters.t = 0;
	gas_out.work_parameters.q = 0;

	Edge *edge;
	ManagerEdgeModelPipeSequential manager;

	// ��������� �������� ���������
	for(int i = 50000; i > 0; --i)
	{
		edge = manager.CreateEdge(&passport);
		edge->set_gas_in(&gas_in);
		edge->set_gas_out(&gas_out);
	}

	// ��������� ������ 100 ��������
	std::cout << "Performing 10 iterations..." << std::endl;
	for(int i = 1; i <= 10; ++i)
	{
		manager.CountAll();
		std::cout << i << " iterations performed" << std::endl;
	}
}


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