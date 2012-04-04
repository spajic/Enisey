#pragma once
#include "math.h" // Для использования abs.

struct GasWorkParameters {
  GasWorkParameters() : p(0), t(0), q(0) { };
  float p;	// Давление, [МПа]
  float t;	// Тепереатура, [К]
  float q;	// Количество газа, [млн м3/сут]
}; 

struct GasCompositionReduced {
  GasCompositionReduced() : density_std_cond(0), n2(0), co2(0) { };
  float density_std_cond;		// Плотность при стд. усл-ях [кг/м3]
  float n2;					// Доля азота [б.р.]
  float co2;					// Доля углекислого газа [б.р.]
};

struct Gas {
  /** Примешать к этому Gas переданный. Результат отражается на этом.
  Смешивание T, rho с.у., CO2, N2 пропорционально объёму. Ну и сам объём.
  Расход может быть отрицательным, если труба реверсивная. Раз потоки всё же
  смешиваются, берём абсолютное значение расходов. */
  void Mix(const Gas &mix) {  
    float q1 = abs(work_parameters.q);
    float q2 = abs(mix.work_parameters.q);
    float sum_q = q1 + q2;
    composition.density_std_cond = // ro1 = (q1*ro1 + q2*ro2) / (q1 + q2)
        (q1 * composition.density_std_cond + 
        q2 * mix.composition.density_std_cond ) / sum_q;
    composition.co2 = 
        (q1 * composition.co2 + q2 * mix.composition.co2 ) / sum_q;
    composition.n2 = (q1 * composition.n2 + q2 * mix.composition.n2 ) / sum_q;
    work_parameters.t = 
        (q1 * work_parameters.t + 
        q2 * mix.work_parameters.t ) / sum_q;   
    /* И в последнюю очередь перерасчитываем q, важно, т.к. предыдущие 
    формулы зависят от work_parameters.q.*/
    if(work_parameters.q > 0) {
      work_parameters.q = sum_q;
    }
    else {
      work_parameters.q = -sum_q;
    }
  }
  GasWorkParameters work_parameters;
  GasCompositionReduced composition;
};