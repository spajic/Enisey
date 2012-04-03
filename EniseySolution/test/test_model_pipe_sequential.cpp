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
  }
  PassportPipe passport;
  ModelPipeSequential pipe;
  Gas gas_in;
  Gas gas_out;
};

/*CountsDirectFlow - Pвх > Pвых, q > 0.*/
TEST_F( ModelPipeSequentialTest, CountsDirectFlow ) {
  pipe.Count();
  EXPECT_NEAR(kTestPipeQuantity, pipe.q(), kTestPipeQuantityPrecision);
}
/*CountsReverseFlow - Pвх < Pвых, q < 0.*/
TEST_F( ModelPipeSequentialTest, CountsReverseFlow ) {
  // Меняем вход и выход местами, получаем реверсивную трубу.
  pipe.set_gas_in(&gas_out); 
  pipe.set_gas_out(&gas_in);
  pipe.Count();
  EXPECT_NEAR(-kTestPipeQuantity, pipe.q(), kTestPipeQuantityPrecision);
} 
