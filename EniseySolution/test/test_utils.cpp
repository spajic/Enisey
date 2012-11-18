/** \file test_utils.cpp
Реализация test_utils.h*/
#include "test_utils.h"
#include "passport_pipe.h"
#include "gas.h"
#include <vector>
#include <iostream>
#include <fstream>
#include "gtest/gtest.h"
#include <list>

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

std::vector<std::string> FileAsVectorOfStrings(std::string filename) {
  std::vector<std::string> res;
  std::ifstream f(filename);
  std::string line;
  while( getline(f, line) ) {
    res.push_back(line);
  }
  return res;
}

/* Проверка совпадения факта и эталона.
  Эталон может быть сформирован в test_gas_transfer_system.*/
void CompareGTSDisbalancesFactToEtalon(
    const std::vector<double> &abs_disbalances,
    const std::vector<int> &int_disbalances) {
  std::ifstream etalon_f;
  etalon_f.open(etalon_saratov_gorkiy_balance);
  if(etalon_f.good() == false) {
    FAIL() << "Can't open etalon file.";
  }
  std::list<double> etalon_abs_disbs;
  std::list<double> etalon_int_disbs;
  int et_iter_num(0);
  double et_abs_d(0.0);
  double et_int_d(0.0);
  while(etalon_f >> et_iter_num >> et_abs_d >> et_int_d) {
    etalon_abs_disbs.push_back(et_abs_d);
    etalon_int_disbs.push_back(et_int_d);
  }
  ASSERT_EQ( abs_disbalances.size(), etalon_abs_disbs.size() );
  EXPECT_TRUE(
      std::equal(
          etalon_abs_disbs.begin(), etalon_abs_disbs.end(),
          abs_disbalances.begin() 
      )
  );
  EXPECT_TRUE(
      std::equal(
          etalon_int_disbs.begin(), etalon_int_disbs.end(),
          int_disbalances.begin() 
      )
  );
}
