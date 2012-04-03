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
  }
  PassportPipe passport;
  ModelPipeSequential pipe;
  Gas gas_in;
  Gas gas_out;
};

/*CountsDirectFlow - P�� > P���, q > 0.*/
TEST_F( ModelPipeSequentialTest, CountsDirectFlow ) {
  pipe.Count();
  EXPECT_NEAR(kTestPipeQuantity, pipe.q(), kTestPipeQuantityPrecision);
}
/*CountsReverseFlow - P�� < P���, q < 0.*/
TEST_F( ModelPipeSequentialTest, CountsReverseFlow ) {
  // ������ ���� � ����� �������, �������� ����������� �����.
  pipe.set_gas_in(&gas_out); 
  pipe.set_gas_out(&gas_in);
  pipe.Count();
  EXPECT_NEAR(-kTestPipeQuantity, pipe.q(), kTestPipeQuantityPrecision);
} 
