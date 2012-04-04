/** \file test_model_pipe_sequential.cpp
Тест класса ModelPipeSequential из test_model_pipe_sequential.h.*/
#include "gtest/gtest.h"
#include "test_utils.h"

/* Тестовый класс, Test-fixture. Всё, что объявлено в нём, доступно тестам.
Создание объектов - в SetUp, освобождение ресурсво - в TearDown.*/
class ModelPipeSequentialTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    passport = MakeTestPassportPipe(); // Создаём пасспорт трубы.
    gas_in = MakeTestGasIn(); // Газ на входе трубы.
    gas_out = MakeTestGasOut(); // Газ на выходе трубы.
    pipe = ModelPipeSequential(&passport); // Создаём и заполняем трубу.
    pipe.set_gas_in(&gas_in);
    pipe.set_gas_out(&gas_out);
    pipe.Count();
  }
  PassportPipe passport;
  ModelPipeSequential pipe;
  Gas gas_in;
  Gas gas_out;
};

/* DirectFlow - Pвх > Pвых, q > 0.*/
TEST_F( ModelPipeSequentialTest, CountsDirectFlow ) {
  EXPECT_NEAR(kTestPipeQuantity, pipe.q(), kTestPipeQuantityPrecision);
}
/* ReverseFlow - Pвх < Pвых, q < 0.*/
TEST_F( ModelPipeSequentialTest, CountsReverseFlow ) {
  // Меняем вход и выход местами, получаем реверсивную трубу.
  pipe.set_gas_in(&gas_out); 
  pipe.set_gas_out(&gas_in);
  pipe.Count();
  EXPECT_NEAR(-kTestPipeQuantity, pipe.q(), kTestPipeQuantityPrecision);
} 
TEST_F( ModelPipeSequentialTest, CountsDerivativesOfProperSigns ) {
  EXPECT_GE( pipe.dq_dp_in(), 0 ); // Растёт давление на входе - растёт расход.
  EXPECT_LE( pipe.dq_dp_out(), 0 );// Растёт P на выходе - расход падает.
}
TEST_F( ModelPipeSequentialTest, CountsDerivativesForReverseFlow ) {
  ModelPipeSequential reverse_pipe = pipe; // Создаём реверсивн ую трубу.
  reverse_pipe.set_gas_in( &gas_out );
  reverse_pipe.set_gas_out( &gas_in );
  reverse_pipe.Count();
  // Производные должны быть обменены местами и знаками.
  EXPECT_FLOAT_EQ(reverse_pipe.dq_dp_in(), -pipe.dq_dp_out() );
  EXPECT_FLOAT_EQ(reverse_pipe.dq_dp_out(), -pipe.dq_dp_in() ); 
}
