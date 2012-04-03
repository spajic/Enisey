#include "test_utils.h"
#include "passport_pipe.h"
#include "gas.h"

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