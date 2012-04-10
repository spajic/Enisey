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
double FindTPseudoCritical(double den_sc, double co2, double n2);
double FindPPseudoCritical(double den_sc, double co2, double n2);
// Фактор сжимаемости при с.у. [б.р.]
double FindZStandartConditions(double den_sc, double co2, double n2);
// Газовая постоянная при с.у. [Джоуль / (Моль*Кельвин)]
double FindRStandartConditions(double den_sc);
// Далее - функции вычисления параметров при рабочих давлении и тем-ре
// Вычислить приведённые давление и температуру
double FindPReduced(double p_work, double p_pseudo_critical);
double FindTReduced(double t_work, double t_pseudo_critical);
// Теплоёмкость при р.у. [Дж/(кг*К)]
double FindC(
    double t_reduced, 
    double p_reduced, 
    double r_standart_conditions);
// Коэффициент Джоуля-Томпсона при рабочих условиях
double FindDi(double p_reduced, double t_reduced);
// Динамическая вязкость при рабочих условиях
double FindMju(double p_reduced, double t_reduced);
// Коэффициент сжимаемости при рабочих условиях [б.р.]
double FindZ(double p_reduced, double t_reduced);
// Плотность при рабочих условиях [кг/м3]
double FindRo(double den_sc, double p_work, double t_work, double z);
// Число Рейнольдса
double FindRe(double q, double den_sc, double mju, double d_inner);
// Коэффициент гидравлического сопротивления (число Лямбда) 
// Требует, чтобы число Рейнольдса было рассчитано!
double FindLambda(
    double re, 
    double d_inner, 
    double roughness_coefficient, 
    double hydraulic_efficiency_coefficient);