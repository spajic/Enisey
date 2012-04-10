#pragma once
/*! \file functions_pipe.h
 *  \brief Функции расчёта трубы.
 *
 *  Более подробное описание файла может быть размещено здесь.
 */
/// \todo Прокоментировать с LaTex-формулами, присести в порядок.
/// \todo Вынести в файл используемые константы.

#include "functions_gas.h"

double ReturnPNextSequential(
    double p_work, 
    double t_work, 
    double q, // рабочие параметры газового потока
    double den_sc, // состав газа
    double r_sc,   // базовые свойства газа          			
    double lambda, 
    double z, // свойства газа при рабочих условиях
    double d_inner, // пасспорт трубы
    double length_of_segment);

double ReturnTNextSequential(
    double p_next, // результат рабботы ReturnPNextSequential
    double p_work, 
    double t_work, 
    double q, // рабочие параметры газового потока
    double den_sc, // состав газа
    double c, 
    double di, // свойства газа при рабочих условиях
    double d_outer, // пасспорт трубы
    double t_env, 
    double heat_exchange_coeff, // свойства окуражающей среды
    double length_of_segment);

// Рассчитать параметры газового потока на выходе трубы
// по свойствам газа, рабочим параметрам на входе трубы, свойствам трубы, св-вам внешней среды, количеству разбиений

void FindSequentialOut(
    double p_work, 
    double t_work, 
    double q,  // рабочие параметры газового потока на входе
    double den_sc, 
    double co2, 
    double n2, // состав газа
    double d_inner, 
    double d_outer, 
    double roughness_coeff, 
    double hydraulic_efficiency_coeff, // св-ва трубы
    double t_env, 
    double heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
    double length_of_segment, 
    int number_of_segments, // длина сегмента и кол-во сегментов
    double* t_out,
    double* p_out);

// Чтобы найти q, нужно решить уравнение Pвых(Pвх, Tвх, q) = p_target относительно q.
// Функция Pвых у нас есть - FindSequentialOut.
// Составим функцию, EquationToSolve(q) = p_target - Pвых(Pвых, Tвх, q).
// Нам нужно найти такое q0, что EquationToSolve(q0) = 0.
// Нужно, чтобы функция по q возрастала
// И чтобы вначале отрезка была отрицательна, проходила через ноль,
// и в конце отрезка была положительна. (Для решения методом деления отрезка пополам).

double EquationToSolve(
    double p_target,
    double p_work, 
    double t_work, 
    double q,  // рабочие параметры газового потока на входе
    double den_sc, 
    double co2, 
    double n2, // состав газа
    double d_inner, 
    double d_outer, 
    double roughness_coeff, 
    double hydraulic_efficiency_coeff, // св-ва трубы
    double t_env, 
    double heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
    double length_of_segment, 
    int number_of_segments, // длина сегмента и кол-во сегментов
    double* t_out, 
    double* p_out);
// Параметры для этой функции - p_target + все те е, что для FindSequentialOut
// за следующим исключением - q - теперь out-параметр, убираем парметр p_out

int FindSequentialQ(
  double p_target, // давление, которое должно получиться в конце
  double p_work, double t_work,  // рабочие параметры газового потока на входе
  double den_sc, double co2, double n2, // состав газа
  double d_inner, double d_outer, double roughness_coeff, double hydraulic_efficiency_coeff, // св-ва трубы
  double t_env, double heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
  double length_of_segment, int number_of_segments, // длина сегмента и кол-во сегментов
  double* t_out, double* q_out);