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
float FindTPseudoCritical(float den_sc, float co2, float n2);
float FindPPseudoCritical(float den_sc, float co2, float n2);
// Фактор сжимаемости при с.у. [б.р.]
float FindZStandartConditions(float den_sc, float co2, float n2);
// Газовая постоянная при с.у. [Джоуль / (Моль*Кельвин)]
float FindRStandartConditions(float den_sc);
// Далее - функции вычисления параметров при рабочих давлении и тем-ре
// Вычислить приведённые давление и температуру
float FindPReduced(float p_work, float p_pseudo_critical);
float FindTReduced(float t_work, float t_pseudo_critical);
// Теплоёмкость при р.у. [Дж/(кг*К)]
float FindC(
    float t_reduced, 
    float p_reduced, 
    float r_standart_conditions);
// Коэффициент Джоуля-Томпсона при рабочих условиях
float FindDi(float p_reduced, float t_reduced);
// Динамическая вязкость при рабочих условиях
float FindMju(float p_reduced, float t_reduced);
// Коэффициент сжимаемости при рабочих условиях [б.р.]
float FindZ(float p_reduced, float t_reduced);
// Плотность при рабочих условиях [кг/м3]
float FindRo(float den_sc, float p_work, float t_work, float z);
// Число Рейнольдса
float FindRe(float q, float den_sc, float mju, float d_inner);
// Коэффициент гидравлического сопротивления (число Лямбда) 
// Требует, чтобы число Рейнольдса было рассчитано!
float FindLambda(
    float re, 
    float d_inner, 
    float roughness_coefficient, 
    float hydraulic_efficiency_coefficient);