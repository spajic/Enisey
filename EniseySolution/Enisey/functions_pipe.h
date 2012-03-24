/*! \file functions_pipe.h
 *  \brief Функции расчёта трубы.
 *
 *  Более подробное описание файла может быть размещено здесь.
 */
#pragma once
#include "math.h"
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
    float length_of_segment) { // длина сегмента
  // static const float kPi = 3.14159265358979323846264338327950288419716939937510;
  // Вычитаемое для расчёта p_next
  float minus = 10*q*q * (length_of_segment) * 
    (16*r_sc * (den_sc*den_sc) * lambda * z * t_work) / 
    (3.1415926535897932384626433832795*3.1415926535897932384626433832795 *
    pow(d_inner/10, 5) );
  float p_next = p_work*p_work - minus;
  // Если полученный квадрат давление больше нуля,
  // возвращаем квадратный корень из него
  if(p_next > 0) {
    return pow(p_next, static_cast<float>(0.5));		 
  }
  // иначе - считаем, что давление упало до нуля.
  else {
    p_next = 0;
  }
  return p_next;
};

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
    float length_of_segment) {
  // kPi = 3.14159265358979323846264338327950288419716939937510
  // коэф-т при (T-Tos) [б.р]
  float s1 = (length_of_segment * d_outer * heat_exchange_coeff * 3.1415926535897932384626433832795) / 
    (c * q * den_sc);
  // Коэф-т при (PNext - P) [К/МПа] 
  float s2 = di*1000000; 
  float t_next = t_work - s1 * (t_work - t_env) + s2 * (p_next - p_work);
  // Если полученная температура меньше температуры окружающей среды, то 
  // считаем, что вышли на температуру окружающей среды
  if(t_next < t_env) {
    t_next = t_env;
  }
  return t_next;
};

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
    float* p_out) { // out - параметры, значения на выходе 
  // В цикле последовательно рассчитываем трубу в "узловых точках"
  float p_next = p_work;
  float t_next = t_work;
  float p_current = p_work;
  float t_current = t_work;

  float p_pseudo_critical = FindPPseudoCritical(den_sc, co2, n2);
  float t_pseudo_critical = FindTPseudoCritical(den_sc, co2, n2);
  float r_sc = FindRStandartConditions(den_sc);

  float p_reduced = 0;
  float t_reduced = 0;
  float z = 0;
  float c = 0;
  float mju = 0;
  float di = 0;
  float re = 0;
  float lambda = 0;

  for(int i = number_of_segments; i != 0; --i) {
    p_current = p_next;
    t_current = t_next;
    // вычисляем необходимые свойства газа при текущих рабочих условиях
    p_reduced = FindPReduced(p_current, p_pseudo_critical);
    t_reduced = FindTReduced(t_current, t_pseudo_critical);

    z = FindZ(p_reduced, t_reduced);
    c = FindC(t_reduced, p_reduced, r_sc);
    di = FindDi(p_reduced, t_reduced);
    mju = FindMju(p_reduced, t_reduced);

    re = FindRe(q, den_sc, mju, d_inner);
    lambda = FindLambda(re, d_inner, roughness_coeff, hydraulic_efficiency_coeff);
    // Вычисляем значения P и T в следующем узле
    p_next = ReturnPNextSequential(p_current, t_current, q, den_sc, r_sc, lambda, z, d_inner, length_of_segment);
    t_next = ReturnTNextSequential(p_next, p_current, t_current, q, den_sc, c, di, d_outer, t_env, heat_exchange_coeff, length_of_segment);
  }

  // запиисываем рассчитанные значения в out-параметры	
  *p_out = p_next;
  *t_out = t_next;
};

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
    float* p_out) {
  FindSequentialOut(
      p_work, 
      t_work, 
      q,  // рабочие параметры газового потока на входе
      den_sc, 
      co2, 
      n2, // состав газа
      d_inner, 
      d_outer, 
      roughness_coeff, 
      hydraulic_efficiency_coeff, // св-ва трубы
      t_env, 
      heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
      length_of_segment, 
      number_of_segments, // длина сегмента и кол-во сегментов
      t_out, 
      p_out); // out - параметры, значения на выходе 
  return p_target - *p_out;
}

// Параметры для этой функции - p_target + все те е, что для FindSequentialOut
// за следующим исключением - q - теперь out-параметр, убираем парметр p_out
int FindSequentialQ(
  float p_target, // давление, которое должно получиться в конце
  float p_work, float t_work,  // рабочие параметры газового потока на входе
  float den_sc, float co2, float n2, // состав газа
  float d_inner, float d_outer, float roughness_coeff, float hydraulic_efficiency_coeff, // св-ва трубы
  float t_env, float heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
  float length_of_segment, int number_of_segments, // длина сегмента и кол-во сегментов
  float* t_out, float* q_out) {// out - параметры, значения на выходе )
  // Решаем уравнение методом деления отрезка пополам
  // Параметры метода - start, finish - определяют отрезок, где ищется решение
  // eps_x, eps_y - точности для условий выхода
  // ToDo: сделать настройки метода параметрами функции.
  // Подумать, как разумно настроить метод деления отрезка пополам для решения 
  // данной задачи.
  // ToDo: корректно обрабатывать возвращаемое значение (для аварийных случаев).
  float start = 0.1;
  float finish = 10000;
  float eps_x = 0.1;
  float eps_y = 0.0001;

  // Заглушка для вызова функции EquationToSolve
  float p_out;
  // Проверки
  // Предполагается, что функция должна возрастать, начинаться ниже нуля, заканчиваться выше нуля.
  if(EquationToSolve(
    p_target,
    p_work, t_work, start,  // рабочие параметры газового потока на входе
    den_sc, co2, n2, // состав газа
    d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // св-ва трубы
    t_env, heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
    length_of_segment, number_of_segments, // длина сегмента и кол-во сегментов
    t_out, &p_out) > 0 )
  {
    //throw "Error. f(start) must be negative";
    return -1;
  }
  if(EquationToSolve(
    p_target,
    p_work, t_work, finish,  // рабочие параметры газового потока на входе
    den_sc, co2, n2, // состав газа
    d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // св-ва трубы
    t_env, heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
    length_of_segment, number_of_segments, // длина сегмента и кол-во сегментов
    t_out, &p_out) < 0)
  {
    //throw "Error. f(finish) must be positive";
    return -2;
  }
  if(start > finish) {
    //throw "Error. Start must be less when finish";
    return -3;
  }

  // local values
  float a = start;
  float b = finish;

  float middle = (a + b) / 2;
  float middle_val = EquationToSolve(
    p_target,
    p_work, t_work, middle,  // рабочие параметры газового потока на входе
    den_sc, co2, n2, // состав газа
    d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // св-ва трубы
    t_env, heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
    length_of_segment, number_of_segments, // длина сегмента и кол-во сегментов
    t_out, &p_out);

  while(abs( middle_val ) > eps_y && abs(a - b) > eps_x) {
    if(middle_val < 0) 
      a = middle;
    else
      b = middle;

    middle = (a + b) / 2;
    middle_val = EquationToSolve(
      p_target,
      p_work, t_work, middle,  // рабочие параметры газового потока на входе
      den_sc, co2, n2, // состав газа
      d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // св-ва трубы
      t_env, heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
      length_of_segment, number_of_segments, // длина сегмента и кол-во сегментов
      t_out, &p_out);
  }

  // Записываем результат в out-параметр.
  *q_out = middle; 

  // Возвращаем код завершения без ошибки.
  return 0;
};