#pragma once

struct GasWorkParameters
{
  GasWorkParameters() : p(0), t(0), q(0) { };
  float p;	// Давление, [МПа]
  float t;	// Тепереатура, [К]
  float q;	// Количество газа, [млн м3/сут]
}; 

struct GasCompositionReduced
{
  GasCompositionReduced() : density_std_cond(0), n2(0), co2(0) { };
  float density_std_cond;		// Плотность при стд. усл-ях [кг/м3]
  float n2;					// Доля азота [б.р.]
  float co2;					// Доля углекислого газа [б.р.]
};

struct Gas
{
  GasWorkParameters work_parameters;
  GasCompositionReduced composition;
};