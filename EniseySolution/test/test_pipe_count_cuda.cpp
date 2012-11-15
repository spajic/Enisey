/** \file test_pipe_count_cuda.cpp
Тест расчёта трубы на CUDA.
Такой же, как test_model_pipe_sequential.cpp */
#include "gtest/gtest.h"
#include "test_utils.h"

#include "passport_pipe.h"
#include "gas.h"

extern "C"
void CountQOnDevice(
  double p_target         ,
  double p_work           , double t_work                     ,
  double den_sc           , double co2                        , double n2,
  double d_inner          , double d_outer                    , 
  double roughness_coeff  , double hydraulic_efficiency_coeff , 
  double t_env            , double heat_exchange_coeff        , 
  double length_of_segment, int number_of_segments            , 
  double* t_out           , double* q_out);

/* Тестовый класс, Test-fixture. Всё, что объявлено в нём, доступно тестам.
Создание объектов - в SetUp, освобождение ресурсво - в TearDown.*/
class PipeCountCUDATest : public ::testing::Test {
protected:
  virtual void SetUp() {
    passport = MakeTestPassportPipe(); // Создаём пасспорт трубы.
    gas_in = MakeTestGasIn(); // Газ на входе трубы.
    gas_out = MakeTestGasOut(); // Газ на выходе трубы.
  }
  PassportPipe passport;
  Gas gas_in;
  Gas gas_out;
};

/* DirectFlow - Pвх > Pвых, q > 0.*/
TEST_F( PipeCountCUDATest, CountsDirectFlow ) {
  int segments = std::max( 1, static_cast<int>(passport.length_)/10 );
  double length_of_segment = passport.length_ / segments;
  double q_out;
  double t_out;
  CountQOnDevice(
    gas_out.work_parameters.p           ,
    gas_in.work_parameters.p            , gas_in.work_parameters.t      ,
    gas_in.composition.density_std_cond , 
    gas_in.composition.co2              , gas_in.composition.n2         ,
    passport.d_inner_                   , passport.d_outer_             , 
    passport.roughness_coeff_ , passport.hydraulic_efficiency_coeff_    , 
    passport.t_env_                     , passport.heat_exchange_coeff_ , 
    length_of_segment                   , segments                      , 
    &t_out                              , &q_out
  );
  EXPECT_NEAR(kTestPipeQuantity, q_out, kTestPipeQuantityPrecision);
}
/* ReverseFlow - Pвх < Pвых, q < 0.*/
TEST_F( PipeCountCUDATest, CountsReverseFlow ) {
  // Меняем вход и выход местами, получаем реверсивную трубу.
  //pipe.set_gas_in(&gas_out); 
  //pipe.set_gas_out(&gas_in);
  //pipe.Count();
  //EXPECT_NEAR(-kTestPipeQuantity, pipe.q(), kTestPipeQuantityPrecision);
} 
TEST_F( PipeCountCUDATest, CountsDerivativesOfProperSigns ) {
  //EXPECT_GE( pipe.dq_dp_in(), 0 ); // Растёт давление на входе - растёт расход.
  //EXPECT_LE( pipe.dq_dp_out(), 0 );// Растёт P на выходе - расход падает.
}
TEST_F( PipeCountCUDATest, CountsDerivativesForReverseFlow ) {
  //ModelPipeSequential reverse_pipe = pipe; // Создаём реверсивн ую трубу.
  //reverse_pipe.set_gas_in( &gas_out );
  //reverse_pipe.set_gas_out( &gas_in );
  //reverse_pipe.Count();
  // Производные должны быть обменены местами и знаками.
  //EXPECT_DOUBLE_EQ(reverse_pipe.dq_dp_in(), -pipe.dq_dp_out() );
  //EXPECT_DOUBLE_EQ(reverse_pipe.dq_dp_out(), -pipe.dq_dp_in() ); 
}
