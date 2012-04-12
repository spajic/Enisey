/** \file test_utils.cpp
Реализация test_utils.h*/
#include "test_utils.h"
#include "passport_pipe.h"
#include "gas.h"
#include <vector>

void FillTestPassportPipe(PassportPipe* passport) {
  passport->d_inner_ = kInnerPipeDiameter;
  passport->d_outer_ = kOuterPipeDiameter;
  passport->length_ = kPipeLength;
  passport->heat_exchange_coeff_ = kHeatExchangeCoefficient;
  passport->hydraulic_efficiency_coeff_ = kHydraulicEfficiencyCoefficient;
  passport->p_max_ = kMaximumPressure;
  passport->p_min_ = kMinimumPressure;
  passport->roughness_coeff_ = kRoughnessCoefficient;
  passport->t_env_ = kEnvironmentTemperature;
}
PassportPipe MakeTestPassportPipe() {
  PassportPipe p;
  FillTestPassportPipe(&p);
  return p;  
}

void FillTestGasIn(Gas* gas) {
  gas->composition.density_std_cond = kInputGasDensityOnStandartConditions;
  gas->composition.co2 = kInputGasCarbonDioxidePart;
  gas->composition.n2 = kInputGasNitrogenPart;
  gas->work_parameters.p = kInputGasPressure;
  gas->work_parameters.t = kInputGasTemperature;
  gas->work_parameters.q = kInputGasQuantity;
}
Gas MakeTestGasIn() {
  Gas g;
  FillTestGasIn(&g);
  return g;
}

void FillTestGasOut(Gas* gas) {
  gas->composition.density_std_cond = kOutputGasDensityOnStandartConditions;
  gas->composition.co2 = kOutputGasCarbonDioxidePart;
  gas->composition.n2 = kOutputGasNitrogenPart;

  gas->work_parameters.p = kOutputGasPressure;
  gas->work_parameters.q = 0;
  gas->work_parameters.t = 0;
}
Gas MakeTestGasOut() {
  Gas g;
  FillTestGasOut(&g);
  return g;
}

/* Всё для составления и решения СЛАУ AX = B
[2 0 1]   [1]   [5]
[0 3 0] * [2] = [6]
[1 2 3]   [3]   [14]
Формат передачи параметров:
A_indexes - индексы (строка*(длина) + столбец) ненулевых коэф-ов А.
A_vals - значения этих коэф-ов в соответствующем порядке.
B - вектор b.
X - вектор найденного решения. 
Так сделано для эффективности передачи разреженных матриц. 
Хотя для них, конечно, есть свои форматы типа CSR.*/
std::vector<int> MakeSimpleSlaeAIndexes() {
  std::vector<int> A_indexes;
  int len = 3; // Длина строки.
  A_indexes.push_back( 0 * len + 0 );  
  A_indexes.push_back( 0 * len + 2 );  
  A_indexes.push_back( 1 * len + 1 );  
  A_indexes.push_back( 2 * len + 0 );  
  A_indexes.push_back( 2 * len + 1 );  
  A_indexes.push_back( 2 * len + 2 );  
  return A_indexes;
} 
std::vector<double> MakeSimpleSlaeAValues() {
  std::vector<double> A_vals;
  A_vals.push_back( 2 );
  A_vals.push_back( 1 );
  A_vals.push_back( 3 );
  A_vals.push_back( 1 );
  A_vals.push_back( 2 );
  A_vals.push_back( 3 );
  return A_vals;
}
std::vector<double> MakeSimpleSlaeB() {
  std::vector<double> b;
  b.push_back(5); b.push_back(6); b.push_back(14);
  return b;
}
std::vector<double> MakeSimpleSlaeX() {
  std::vector<double> x;
  x.push_back(1); x.push_back(2); x.push_back(3);
  return x;
}