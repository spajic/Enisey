/** \file test_model_pipe_sequential.cpp
���� ������ ModelPipeSequential �� test_model_pipe_sequential.h.*/
#include "gtest/gtest.h"
#include "test_utils.h"

/* �������� �����, Test-fixture. ��, ��� ��������� � ��, �������� ������.
�������� �������� - � SetUp, ������������ �������� - � TearDown.*/
class ModelPipeSequentialTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    passport = MakeTestPassportPipe(); // ������ �������� �����.
    gas_in = MakeTestGasIn(); // ��� �� ����� �����.
    gas_out = MakeTestGasOut(); // ��� �� ������ �����.
    pipe = ModelPipeSequential(&passport); // ������ � ��������� �����.
    pipe.set_gas_in(&gas_in);
    pipe.set_gas_out(&gas_out);
    pipe.Count();
  }
  PassportPipe passport;
  ModelPipeSequential pipe;
  Gas gas_in;
  Gas gas_out;
};

/* DirectFlow - P�� > P���, q > 0.*/
TEST_F( ModelPipeSequentialTest, CountsDirectFlow ) {
  EXPECT_NEAR(kTestPipeQuantity, pipe.q(), kTestPipeQuantityPrecision);
}
/* ReverseFlow - P�� < P���, q < 0.*/
TEST_F( ModelPipeSequentialTest, CountsReverseFlow ) {
  // ������ ���� � ����� �������, �������� ����������� �����.
  pipe.set_gas_in(&gas_out); 
  pipe.set_gas_out(&gas_in);
  pipe.Count();
  EXPECT_NEAR(-kTestPipeQuantity, pipe.q(), kTestPipeQuantityPrecision);
} 
TEST_F( ModelPipeSequentialTest, CountsDerivativesOfProperSigns ) {
  EXPECT_GE( pipe.dq_dp_in(), 0 ); // ����� �������� �� ����� - ����� ������.
  EXPECT_LE( pipe.dq_dp_out(), 0 );// ����� P �� ������ - ������ ������.
}
TEST_F( ModelPipeSequentialTest, CountsDerivativesForReverseFlow ) {
  ModelPipeSequential reverse_pipe = pipe; // ������ ��������� �� �����.
  reverse_pipe.set_gas_in( &gas_out );
  reverse_pipe.set_gas_out( &gas_in );
  reverse_pipe.Count();
  // ����������� ������ ���� �������� ������� � �������.
  EXPECT_DOUBLE_EQ(reverse_pipe.dq_dp_in(), -pipe.dq_dp_out() );
  EXPECT_DOUBLE_EQ(reverse_pipe.dq_dp_out(), -pipe.dq_dp_in() ); 
}
