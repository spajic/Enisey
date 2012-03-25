#include "test_utils.h"
#include "passport_pipe.h"
#include "gas.h"

void FillTestPassportPipe(PassportPipe* passport)
{
  passport->d_inner_ = 1000;
  passport->d_outer_ = 1020;
  passport->length_ = 100;
  passport->heat_exchange_coeff_ = 1.3;
  passport->hydraulic_efficiency_coeff_ = 0.95;
  passport->p_max_ = 100;
  passport->p_min_ = 1;
  passport->roughness_coeff_ = 0.03;
  passport->t_env_ = 280.15;
}

void FillTestGasIn(Gas* gas)
{
  gas->composition.density_std_cond = 0.6865365; // [кг/м3]
  gas->composition.co2 = 0;
  gas->composition.n2 = 0;

  gas->work_parameters.p = 5; // [МПа]
  gas->work_parameters.t = 293.15; // [К]
  gas->work_parameters.q = 387.843655734; // [м3/сек]
}

void FillTestGasOut(Gas* gas)
{
  gas->composition.density_std_cond = 0.6865365; // [кг/м3]
  gas->composition.co2 = 0;
  gas->composition.n2 = 0;

  gas->work_parameters.p = 3;
  gas->work_parameters.q = 0;
  gas->work_parameters.t = 0;
}