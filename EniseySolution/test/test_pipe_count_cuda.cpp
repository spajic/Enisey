/** \file test_pipe_count_cuda.cpp
Тест расчёта трубы на CUDA.
Такой же, как test_model_pipe_sequential.cpp */
#include "gtest/gtest.h"
#include "test_utils.h"

#include "passport_pipe.h"
#include "work_params.h"
#include "calculated_params.h"
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
// Пока до конца не решил, что должно в этом случае происходить.
// Новый подход: модели работают только с обычными трубами, 
// Отслеживать и управлять случаями реверсивных труб дело того, кто
// формирует данные - суперменеджера, например.
// Это позволит упростить логику всех моделей и сосредоточить управление
// этим нюансом в одном месте - в суперменеджере.
// А моделям, по идее, следует выбрасывать каким-то образом сообщение об
// ошибке - недопустимые входные данные.
TEST_F( PipeCountCUDATest, CountsReverseFlow ) {
  // Меняем вход и выход местами, получаем реверсивную трубу.
  /*int segments = std::max( 1, static_cast<int>(passport.length_)/10 );
  double length_of_segment = passport.length_ / segments;
  double q_out;
  double t_out;
  CountQOnDevice(
    gas_in.work_parameters.p            ,
    gas_out.work_parameters.p           , gas_out.work_parameters.t     ,
    gas_out.composition.density_std_cond, 
    gas_out.composition.co2             , gas_out.composition.n2        ,
    passport.d_inner_                   , passport.d_outer_             , 
    passport.roughness_coeff_ , passport.hydraulic_efficiency_coeff_    , 
    passport.t_env_                     , passport.heat_exchange_coeff_ , 
    length_of_segment                   , segments                      , 
    &t_out                              , &q_out
    );
  EXPECT_NEAR(-kTestPipeQuantity, q_out, kTestPipeQuantityPrecision);*/
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

/* Проверка, что расчёт первой трубы эталона, построенной по схеме Саратова
даёт такой же результат, как эталонный расчёт на CPU. */
TEST( PipeCountCUDATestSaratov, CountsFirstPipeOfSaratov ) {
  PassportPipe pass;
  pass.d_inner_                     = 515;
  pass.d_outer_                     = 530;
  pass.heat_exchange_coeff_         = 1.3;
  pass.hydraulic_efficiency_coeff_  = 0.7007;
  pass.length_                      = 4;
  pass.p_max_                       = 5.3936;
  pass.p_min_                       = 1.96133;
  pass.roughness_coeff_             = 0.03;
  pass.t_env_                       = 275.65;

  WorkParams wp;
  wp.set_p_in     (2.2880811667883729);
  wp.set_p_out    (2.2596797626698923);
  wp.set_t_in     (278.25498874278276);
  wp.set_den_sc_in(0.68991000000000002);
  wp.set_n2_in    (0);
  wp.set_co2_in   (0);
  int segments = std::max( 1, static_cast<int>(pass.length_)/10 );
  double length_of_segment = pass.length_ / segments;
  double q_out;
  double t_out;
  CountQOnDevice(
    wp.p_out()            ,
    wp.p_in()             , wp.t_in()                       ,
    wp.den_sc_in()        , 
    wp.co2_in()           , wp.n2_in()                      ,
    pass.d_inner_         , pass.d_outer_                   , 
    pass.roughness_coeff_ , pass.hydraulic_efficiency_coeff_, 
    pass.t_env_           , pass.heat_exchange_coeff_       , 
    length_of_segment     , segments                        , 
    &t_out                , &q_out
    );
  CalculatedParams etalon;
  etalon.set_q(22.12012700520966);
  etalon.set_t_out(277.4648998599593);
  EXPECT_NEAR(etalon.q()    , q_out, kTestPipeQuantityPrecision);
  EXPECT_NEAR(etalon.t_out(), t_out, kTestPipeQuantityPrecision);
}