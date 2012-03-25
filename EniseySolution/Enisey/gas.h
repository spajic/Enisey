#pragma once

struct GasWorkParameters
{
  float p;	// Давление, [МПа]
  float t;	// Тепереатура, [К]
  float q;	// Количество газа, [млн м3/сут]
}; 

struct GasCompositionReduced
{
  float density_std_cond;		// Плотность при стд. усл-ях [кг/м3]
  float n2;					// Доля азота [б.р.]
  float co2;					// Доля углекислого газа [б.р.]
};

struct Gas
{
  GasWorkParameters work_parameters;
  GasCompositionReduced composition;
};