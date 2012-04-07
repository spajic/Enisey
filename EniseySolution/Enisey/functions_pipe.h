#pragma once
/*! \file functions_pipe.h
 *  \brief Функции расчёта трубы.
 *
 *  Более подробное описание файла может быть размещено здесь.
 */
/// \todo Прокоментировать с LaTex-формулами, присести в порядок.
/// \todo Вынести в файл используемые константы.

#include "functions_gas.h"

float ReturnPNextSequential(
    float p_work, 
    float t_work, 
    float q, // рабочие параметры газового потока
    float den_sc, // состав газа
    float r_sc,   // базовые свойства газа          			
    float lambda, 
    float z, // свойства газа при рабочих условиях
    float d_inner, // пасспорт трубы
    float length_of_segment);

float ReturnTNextSequential(
    float p_next, // результат рабботы ReturnPNextSequential
    float p_work, 
    float t_work, 
    float q, // рабочие параметры газового потока
    float den_sc, // состав газа
    float c, 
    float di, // свойства газа при рабочих условиях
    float d_outer, // пасспорт трубы
    float t_env, 
    float heat_exchange_coeff, // свойства окуражающей среды
    float length_of_segment);

// Рассчитать параметры газового потока на выходе трубы
// по свойствам газа, рабочим параметрам на входе трубы, свойствам трубы, св-вам внешней среды, количеству разбиений

void FindSequentialOut(
    float p_work, 
    float t_work, 
    float q,  // рабочие параметры газового потока на входе
    float den_sc, 
    float co2, 
    float n2, // состав газа
    float d_inner, 
    float d_outer, 
    float roughness_coeff, 
    float hydraulic_efficiency_coeff, // св-ва трубы
    float t_env, 
    float heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
    float length_of_segment, 
    int number_of_segments, // длина сегмента и кол-во сегментов
    float* t_out,
    float* p_out);

// Чтобы найти q, нужно решить уравнение Pвых(Pвх, Tвх, q) = p_target относительно q.
// Функция Pвых у нас есть - FindSequentialOut.
// Составим функцию, EquationToSolve(q) = p_target - Pвых(Pвых, Tвх, q).
// Нам нужно найти такое q0, что EquationToSolve(q0) = 0.
// Нужно, чтобы функция по q возрастала
// И чтобы вначале отрезка была отрицательна, проходила через ноль,
// и в конце отрезка была положительна. (Для решения методом деления отрезка пополам).

float EquationToSolve(
    float p_target,
    float p_work, 
    float t_work, 
    float q,  // рабочие параметры газового потока на входе
    float den_sc, 
    float co2, 
    float n2, // состав газа
    float d_inner, 
    float d_outer, 
    float roughness_coeff, 
    float hydraulic_efficiency_coeff, // св-ва трубы
    float t_env, 
    float heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
    float length_of_segment, 
    int number_of_segments, // длина сегмента и кол-во сегментов
    float* t_out, 
    float* p_out);
// Параметры для этой функции - p_target + все те е, что для FindSequentialOut
// за следующим исключением - q - теперь out-параметр, убираем парметр p_out

int FindSequentialQ(
  float p_target, // давление, которое должно получиться в конце
  float p_work, float t_work,  // рабочие параметры газового потока на входе
  float den_sc, float co2, float n2, // состав газа
  float d_inner, float d_outer, float roughness_coeff, float hydraulic_efficiency_coeff, // св-ва трубы
  float t_env, float heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
  float length_of_segment, int number_of_segments, // длина сегмента и кол-во сегментов
  float* t_out, float* q_out);