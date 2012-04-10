/** \file test_functions_gas.cpp
Тесты для функций расчёта свойств газа из functions_gas.h*/
#include "gtest/gtest.h"
#include "test_utils.h"

#include "model_pipe_sequential.h"
#include "gas.h"
#include "functions_pipe.h"
#include "passport_pipe.h"
#include "edge.h"
#include "manager_edge_model_pipe_sequential.h"
#include "loader_vesta.h"
TEST(Gas, GasCountFunctions) {
  // Тестируем правильность работы газовых функций.
  // Тест такой - в Весте выставяляем использование НТП-2006
  // В локальной задаче смотрим вычисленные Вестой значения.
  // Если получается довольно похоже, принимаем рассчитанные мной значения
  // за эталон - и сохраняем для дальнейшего сравнения с ними.
  // В Весте показываются значения для z, c, mju.

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
  double t_pseudo_critical = FindTPseudoCritical(
    composition.density_std_cond, composition.co2, composition.n2);
  double p_pseudo_critical = FindPPseudoCritical(
    composition.density_std_cond, composition.co2, composition.n2);
  double z_standart_conditions = FindZStandartConditions(
    composition.density_std_cond, composition.co2, composition.n2);
  double r_standart_conditions = FindRStandartConditions(
    composition.density_std_cond);

  // Вычисление характеристик газа при рабочем давлении и температуре
  // p_reduced = 0.64219296
  // t_reduced = 1.4416289
  // c = 2372.5037		(значение в Весте Cp = 2310.1)
  // di = 5.0115082e-6
  // mju = 1.0930206e-5	(значение в Весте mju = 1.093e-5)
  // z = 0.91878355		(значение в Весте z = 0.91878)
  // ro = 23.002302
  double p_reduced = FindPReduced(params.p, p_pseudo_critical);
  double t_reduced = FindTReduced(params.t, t_pseudo_critical);
  double c = FindC(t_reduced, p_reduced, r_standart_conditions);
  double di = FindDi(p_reduced, t_reduced);
  double mju = FindMju(p_reduced, t_reduced);
  double z = FindZ(p_reduced, t_reduced);
  double ro = FindRo(composition.density_std_cond, params.p, params.t, z);

  /* Проверим на адекватность так же функции, 
  которые зависят и от паспорта трубы*/
  PassportPipe passport;
  FillTestPassportPipe(&passport);

  // re = 31017164.0
  // lambda = 0.010797811
  double re = FindRe(params.q, composition.density_std_cond, mju, passport.d_inner_);
  double lambda = FindLambda(re, passport.d_inner_, passport.roughness_coeff_, passport.hydraulic_efficiency_coeff_);

  // Видно, что значения mju и z практически совпали с Вестой,
  // с немного отличается. 
  // Принимаем вычисленные решения за эталон и делаем EXPECT'ы
  // Если значения вдруг станут вычислятся по другому, тест даст нам знать.

  // Задаём точность сравнения eps
  double eps = 1.0e-4;

  // Проверки базовых параметров
  EXPECT_LE(abs(t_pseudo_critical - 194.07213), eps);
  EXPECT_LE(abs(p_pseudo_critical - 4.6355228), eps);
  EXPECT_LE(abs(z_standart_conditions - 0.99798650), eps);
  EXPECT_LE(abs(r_standart_conditions - 503.37848), eps);

  // Проверки параметров при рабочих условиях
  EXPECT_LE(abs(p_reduced - 0.64219296), eps);
  EXPECT_LE(abs(t_reduced - 1.4416289), eps);
  EXPECT_LE(abs(c - 2372.5037), eps);
  EXPECT_LE(abs(di - 5.0115082e-6), eps);
  EXPECT_LE(abs(mju - 1.0930206e-5), eps);
  EXPECT_LE(abs(z - 0.91878355), eps);
  EXPECT_LE(abs(ro - 23.002302), eps);

  // Проверки для re и lambda
  // Для re задаём свою точность, потому что число большое
  double eps_re = 1;
  EXPECT_LE(abs(re - 31017164.0), eps_re);
  EXPECT_LE(abs(lambda - 0.010797811), eps);
}