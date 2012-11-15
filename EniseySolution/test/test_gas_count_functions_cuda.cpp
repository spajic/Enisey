/** \file test_gas_count_functions_cuda.cpp
Тесты для функций расчёта свойств газа, работающих на CUDA
из файла gas_count_functions_cuda.cuh
Тест такой же, как и test_functions_gas.cpp.
*/
#include "gtest/gtest.h"
#include "test_utils.h"

//#include "gas_count_functions_cuda.cuh"

extern "C"
  void FindBasicGasPropsOnDevice(
  double den_sc, double co2, double n2,
  double *t_pc_out, double *p_pc_out, double *z_sc_out, double *r_sc_out);

extern "C"
  void FindGasPropsAtWorkParamsOnDevice(
  double p_work,      double t_work, 
  double p_pc,        double t_pc, 
  double den_sc,      double r_sc,
  double *p_reduced,  double *t_reduced, 
  double *c,          double *di, 
  double *mju,        double *z, 
  double *ro);

extern "C" 
  void FindReAndLambdaOnDevice(
  double q    , double den_sc ,
  double mju  , double d_inner,
  double rough, double hydr   ,
  double *re  , double *lambda);

TEST(Gas, GasCountFunctionsCUDA) {
  // Тестируем правильность работы газовых функций.
  // Тест такой - в Весте выставяляем использование НТП-2006
  // В локальной задаче смотрим вычисленные Вестой значения.
  // Если получается довольно похоже, принимаем рассчитанные мной значения
  // за эталон - и сохраняем для дальнейшего сравнения с ними.
  // В Весте показываются значения для z, c, mju.

  // Задаём точность сравнения eps
  double eps = 1.0e-4;

  // Задаём свойства газа, для которого проводим тестирование.
  GasCompositionReduced composition;
  composition.density_std_cond = 0.6865365; // [кг/м3]
  composition.co2 = 0;
  composition.n2 = 0;
  GasWorkParameters params;
  params.p = 2.9769; // [МПа]
  params.t = 279.78; // [К]
  params.q = 387.843655734; // [м3/сек]
  // Вычисление базовых характеристик газа только по составу
  // t_pseudo_critical = 194.07213 [К]
  // p_pseudo_critical = 4.6355228 [МПа]
  // z_standart_conditions = 0.99798650 [б.р.]
  // r_standart_conditions = 503.37848 [Дж / (кг * К)]
  double t_pseudo_critical;
  double p_pseudo_critical;
  double z_standart_conditions;
  double r_standart_conditions;
  FindBasicGasPropsOnDevice(
      composition.density_std_cond, composition.co2, composition.n2,
      &t_pseudo_critical, &p_pseudo_critical, 
      &z_standart_conditions, &r_standart_conditions);
  // Проверки базовых параметров
  EXPECT_LE(abs(t_pseudo_critical - 194.07213), eps);
  EXPECT_LE(abs(p_pseudo_critical - 4.6355228), eps);
  EXPECT_LE(abs(z_standart_conditions - 0.99798650), eps);
  EXPECT_LE(abs(r_standart_conditions - 503.37848), eps);

  // Вычисление характеристик газа при рабочем давлении и температуре
  // p_reduced = 0.64219296
  // t_reduced = 1.4416289
  // c = 2372.5037		(значение в Весте Cp = 2310.1)
  // di = 5.0115082e-6
  // mju = 1.0930206e-5	(значение в Весте mju = 1.093e-5)
  // z = 0.91878355		(значение в Весте z = 0.91878)
  // ro = 23.002302
  double p_reduced;
  double t_reduced;
  double c;
  double di;
  double mju;
  double z;
  double ro;
  FindGasPropsAtWorkParamsOnDevice(
    params.p                      , params.t              , 
    p_pseudo_critical             , t_pseudo_critical     , 
    composition.density_std_cond  , r_standart_conditions ,
    &p_reduced                    , &t_reduced            , 
    &c                            , &di                   , 
    &mju                          , &z                    , 
    &ro);
  // Проверки параметров при рабочих условиях
  EXPECT_LE(abs(p_reduced - 0.64219296), eps);
  EXPECT_LE(abs(t_reduced - 1.4416289), eps);
  EXPECT_LE(abs(c - 2372.5037), eps);
  EXPECT_LE(abs(di - 5.0115082e-6), eps);
  EXPECT_LE(abs(mju - 1.0930206e-5), eps);
  EXPECT_LE(abs(z - 0.91878355), eps);
  EXPECT_LE(abs(ro - 23.002302), eps);
  // Видно, что значения mju и z практически совпали с Вестой,
  // с немного отличается. 
  // Принимаем вычисленные решения за эталон и делаем EXPECT'ы
  // Если значения вдруг станут вычислятся по другому, тест даст нам знать.

  /* Проверим на адекватность так же функции, 
  которые зависят и от паспорта трубы*/
  PassportPipe passport = MakeTestPassportPipe();
  // re = 31017164.0
  // lambda = 0.010797811
  double re;
  double lambda;  
  FindReAndLambdaOnDevice(
    params.q                  , composition.density_std_cond        ,
    mju                       , passport.d_inner_                   ,
    passport.roughness_coeff_ , passport.hydraulic_efficiency_coeff_,
    &re                       , &lambda);
  // Проверки для re и lambda
  // Для re задаём свою точность, потому что число большое
  double eps_re = 1;
  EXPECT_LE(abs(re - 31017165.308), eps_re);
  EXPECT_LE(abs(lambda - 0.010797811), eps);
}