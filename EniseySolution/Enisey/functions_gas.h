#pragma once
/*! \file functions_gas.h
 *  \brief Функции расчёта различных свойств газа.
 *
 *  Более подробное описание файла может быть размещено здесь.
 */
/// \todo Прокоментировать с LaTex-формулами.
/** \todo Вынести куда-нибудь на отдельную м.б. страницу в документации
мой подход к размерностям. Какие величины в каких размерностях.*/
/** Псевдо-критическая температура газа [К]
\f$T_{пк}=88.25 * (0.9915 + 1.759 * \rho_{с.у.} - CO_2 - 1.681*N_2)\f$
\param den_sc Плотность при стандартных условиях [кг/м3]
\param co2 Доля содержания CO2 [б.р.]
\param n2 Доля содержания N2 [б.р.] */
inline float FindTPseudoCritical(float den_sc, float co2, float n2) {
  return 88.25 * (0.9915 + 1.759 * den_sc - co2 - 1.681*n2); // [Кельвины]
}
inline float FindPPseudoCritical(float den_sc, float co2, float n2) {
  return 2.9585 * (1.608 - 0.05994*den_sc - co2 - 0.392*n2); // [МегаПаскали]
};
// Фактор сжимаемости при с.у. [б.р.]
inline float FindZStandartConditions(float den_sc, float co2, float n2) {
  return 1 - (0.0741*den_sc-0.006-0.063*n2-0.0575*co2) * 
    (0.0741*den_sc-0.006-0.063*n2-0.0575*co2); // [б.р.]
};
// Газовая постоянная при с.у. [Джоуль / (Моль*Кельвин)]
inline float FindRStandartConditions(float den_sc) {
  // kAirDensityStandartConditions = 1.2046
  return 286.89 * 1.2046 / den_sc; // [Дж/(моль*К)]
};
// Далее - функции вычисления параметров при рабочих давлении и тем-ре
// Вычислить приведённые давление и температуру
inline float FindPReduced(float p_work, float p_pseudo_critical) {
  return p_work / p_pseudo_critical; // [б.р.]
};
inline float FindTReduced(float t_work, float t_pseudo_critical) {
  return t_work / t_pseudo_critical; // [б.р.]
};
// Теплоёмкость при р.у. [Дж/(кг*К)]
inline float FindC(
    float t_reduced, 
    float p_reduced, 
    float r_standart_conditions) {
  float E0 = 4.437 - 1.015*t_reduced + 0.591*t_reduced*t_reduced;
  float E1 = 3.29  - 11.37/t_reduced + 10.9 /(t_reduced*t_reduced);
  float E2 = 3.23  - 16.27/t_reduced + 25.48/(t_reduced*t_reduced) 
    - 11.81/(t_reduced*t_reduced*t_reduced);
  float E3 = -0.214 + 0.908/t_reduced -0.967/(t_reduced*t_reduced);

  return r_standart_conditions * (E0 + E1*p_reduced + E2*(p_reduced*p_reduced) + 
    E3*(p_reduced*p_reduced*p_reduced) ); // [Дж/(кг*К)]
};
// Коэффициент Джоуля-Томпсона при рабочих условиях
inline float FindDi(float p_reduced, float t_reduced) {
  float H0 = 24.94 - 20.3*t_reduced  + 4.57  *(t_reduced*t_reduced);
  float H1 = 5.66  - 19.92/t_reduced + 16.89 /(t_reduced*t_reduced);
  float H2 = -4.11 + 14.68/t_reduced - 13.39 /(t_reduced*t_reduced);
  float H3 = 0.568 - 2.0/t_reduced   + 1.79  /(t_reduced*t_reduced);

  float di_ = H0 + H1*p_reduced + H2*(p_reduced*p_reduced) + 
    H3*(p_reduced*p_reduced*p_reduced); 
  di_ *= 0.000001; // ToDo: разбраться с этим множителем
  return di_;
};
// Динамическая вязкость при рабочих условиях
inline float FindMju(float p_reduced, float t_reduced) {
  float Mju0 = (1.81 + 5.95*t_reduced); 
  float B1 = -0.67 + 2.36/t_reduced  - 1.93  / (t_reduced*t_reduced);
  float B2 = 0.8   - 2.89/t_reduced  + 2.65  / (t_reduced*t_reduced);
  float B3 = -0.1  + 0.354/t_reduced - 0.314 / (t_reduced*t_reduced);

  float mju_ = Mju0 * (1 + B1*p_reduced + B2*(p_reduced*p_reduced) + 
    B3*(p_reduced*p_reduced*p_reduced));
  mju_ /= 1000000; // ToDo: разобраться с этим множителем
  return mju_;
};
// Коэффициент сжимаемости при рабочих условиях [б.р.]
inline float FindZ(float p_reduced, float t_reduced) {
  float A1 = -0.39 + 2.03/t_reduced - 3.16/(t_reduced*t_reduced) 
    + 1.09/(t_reduced*t_reduced*t_reduced);
  float A2 = 0.0423 - 0.1812/t_reduced + 0.2124/(t_reduced*t_reduced);
  return 1 + A1*p_reduced + A2*(p_reduced*p_reduced);
};
// Плотность при рабочих условиях [кг/м3]
inline float FindRo(float den_sc, float p_work, float t_work, float z) {
  //static const float kTemperatureStandartConditions = 293.15; // [K] 
  //static const float  kPressureStandartConditions = 0.101325; // [MПа] 
  return (den_sc * 293.15 * p_work) / (t_work*0.101325*z);
};	
// Число Рейнольдса
inline float FindRe(float q, float den_sc, float mju, float d_inner) {
  // kPi = 3.14159265358979323846264338327950288419716939937510
  return (4.0/3.14159265358979323846264338327950288419716939937510) * (q*den_sc) / (mju*(d_inner/1000));
};
// Коэффициент гидравлического сопротивления (число Лямбда) 
// Требует, чтобы число Рейнольдса было рассчитано!
inline float FindLambda(
    float re, 
    float d_inner, 
    float roughness_coefficient, 
    float hydraulic_efficiency_coefficient) {
  float osn = 158.0/re + (2*roughness_coefficient/d_inner);
  float lambda = 0.067 * exp(0.2 * log(osn)); 
  // ToDo: возможно стоит хранить сразу квадрат к-та гидравлич. эф-ти
  lambda /= pow(hydraulic_efficiency_coefficient, 2); 
  return lambda;
};