// Содержание этого файла теперь в functions_pipe.h
/*
#pragma once

#include "math.h"

//#include "gas_count_functions.cuh" - теперь в functions_gas.h
#include "functions_gas.h"

double ReturnPNextSequential(
	double p_work, double t_work, double q, // рабочие параметры газового потока
	double den_sc, // состав газа
	double r_sc,   // базовые свойства газа          			
	double lambda, double z, // свойства газа при рабочих условиях
	double d_inner, // пасспорт трубы
	double length_of_segment) // длина сегмента
{
	// static const double kPi = 3.14159265358979323846264338327950288419716939937510;
	// Вычитаемое для расчёта p_next
	double minus = 10*q*q * (length_of_segment) * 
		(16*r_sc * (den_sc*den_sc) * lambda * z * t_work) / 
		(3.1415926535897932384626433832795*3.1415926535897932384626433832795 *
		pow(d_inner/10, 5) );
	double p_next = p_work*p_work - minus;
	// Если полученный квадрат давление больше нуля,
	// возвращаем квадратный корень из него
	if(p_next > 0)
	{
		return pow(p_next, static_cast<double>(0.5));		 
	}
	// иначе - считаем, что давление упало до нуля.
	else
	{
		p_next = 0;
	}
	return p_next;
};

double ReturnTNextSequential(
	double p_next, // результат рабботы ReturnPNextSequential
	double p_work, double t_work, double q, // рабочие параметры газового потока
	double den_sc, // состав газа
	double c, double di, // свойства газа при рабочих условиях
	double d_outer, // пасспорт трубы
	double t_env, double heat_exchange_coeff, // свойства окуражающей среды
	double length_of_segment)
{
	// kPi = 3.14159265358979323846264338327950288419716939937510
	// коэф-т при (T-Tos) [б.р]
	double s1 = (length_of_segment * d_outer * heat_exchange_coeff * 3.1415926535897932384626433832795) / 
		(c * q * den_sc);

	// Коэф-т при (PNext - P) [К/МПа] 
	double s2 = di*1000000; 
	double t_next = t_work - s1 * (t_work - t_env) + s2 * (p_next - p_work);
	// Если полученная температура меньше температуры окружающей среды, то 
	// считаем, что вышли на температуру окружающей среды
	if(t_next < t_env)
	{
		t_next = t_env;
	}
	return t_next;
};

// Рассчитать параметры газового потока на выходе трубы
// по свойствам газа, рабочим параметрам на входе трубы, свойствам трубы, св-вам внешней среды, количеству разбиений
void FindSequentialOut(
	double p_work, double t_work, double q,  // рабочие параметры газового потока на входе
	double den_sc, double co2, double n2, // состав газа
	double d_inner, double d_outer, double roughness_coeff, double hydraulic_efficiency_coeff, // св-ва трубы
	double t_env, double heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
	double length_of_segment, int number_of_segments, // длина сегмента и кол-во сегментов
	double* t_out, double* p_out) // out - параметры, значения на выходе 
{
	// В цикле последовательно рассчитываем трубу в "узловых точках"
	double p_next = p_work;
	double t_next = t_work;
	double p_current = p_work;
	double t_current = t_work;

	double p_pseudo_critical = FindPPseudoCritical(den_sc, co2, n2);
	double t_pseudo_critical = FindTPseudoCritical(den_sc, co2, n2);
	double r_sc = FindRStandartConditions(den_sc);

	double p_reduced = 0;
	double t_reduced = 0;
	double z = 0;
	double c = 0;
	double mju = 0;
	double di = 0;
	double re = 0;
	double lambda = 0;
		
	for(int i = number_of_segments; i != 0; --i)
	{
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
double EquationToSolve(
	double p_target,
	double p_work, double t_work, double q,  // рабочие параметры газового потока на входе
	double den_sc, double co2, double n2, // состав газа
	double d_inner, double d_outer, double roughness_coeff, double hydraulic_efficiency_coeff, // св-ва трубы
	double t_env, double heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
	double length_of_segment, int number_of_segments, // длина сегмента и кол-во сегментов
	double* t_out, double* p_out)
{
	FindSequentialOut(
		p_work, t_work, q,  // рабочие параметры газового потока на входе
		den_sc, co2, n2, // состав газа
		d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // св-ва трубы
		t_env, heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
		length_of_segment, number_of_segments, // длина сегмента и кол-во сегментов
		t_out, p_out); // out - параметры, значения на выходе 
	return p_target - *p_out;
}

// Параметры для этой функции - p_target + все те е, что для FindSequentialOut
// за следующим исключением - q - теперь out-параметр, убираем парметр p_out
int FindSequentialQ(
	double p_target, // давление, которое должно получиться в конце
	double p_work, double t_work,  // рабочие параметры газового потока на входе
	double den_sc, double co2, double n2, // состав газа
	double d_inner, double d_outer, double roughness_coeff, double hydraulic_efficiency_coeff, // св-ва трубы
	double t_env, double heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
	double length_of_segment, int number_of_segments, // длина сегмента и кол-во сегментов
	double* t_out, double* q_out) // out - параметры, значения на выходе )
{
	// Решаем уравнение методом деления отрезка пополам
	// Параметры метода - start, finish - определяют отрезок, где ищется решение
	// eps_x, eps_y - точности для условий выхода
	// ToDo: сделать настройки метода параметрами функции.
	// Подумать, как разумно настроить метод деления отрезка пополам для решения 
	// данной задачи.
	// ToDo: корректно обрабатывать возвращаемое значение (для аварийных случаев).
	double start = 0.1;
	double finish = 10000;
	double eps_x = 0.1;
	double eps_y = 0.0001;

	// Заглушка для вызова функции EquationToSolve
	double p_out;
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
	if(start > finish)
	{
		//throw "Error. Start must be less when finish";
		return -3;
	}

	// local values
	double a = start;
	double b = finish;

	double middle = (a + b) / 2;
	double middle_val = EquationToSolve(
		p_target,
		p_work, t_work, middle,  // рабочие параметры газового потока на входе
		den_sc, co2, n2, // состав газа
		d_inner, d_outer, roughness_coeff, hydraulic_efficiency_coeff, // св-ва трубы
		t_env, heat_exchange_coeff, // св-ва внешней среды (тоже входят в пасспорт трубы)
		length_of_segment, number_of_segments, // длина сегмента и кол-во сегментов
		t_out, &p_out);

	while(abs( middle_val ) > eps_y && abs(a - b) > eps_x)
	{
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
};*/