/** \file model_pipe_sequential.cpp
Реализация model_pipe_sequential.h*/
#include "model_pipe_sequential.h"
#include "functions_pipe.h" // Функции расчёта свойств газа.

ModelPipeSequential::ModelPipeSequential() {

}
ModelPipeSequential::ModelPipeSequential(const Passport* passport) {
	passport_ = *(dynamic_cast<const PassportPipe*>(passport));
}

void ModelPipeSequential::set_gas_in(const Gas* gas) {
	gas_in_ = *gas;
}
const Gas& ModelPipeSequential::gas_in() {
  return gas_in_;
}
void ModelPipeSequential::set_gas_out(const Gas* gas) {
	gas_out_ = *gas;
}
const Gas& ModelPipeSequential::gas_out() {
  return gas_out_;
}
PassportPipe* ModelPipeSequential::passport() {
  return &passport_;
}

inline
bool ModelPipeSequential::IsReverse() {
  return (gas_in_.work_parameters.p < gas_out_.work_parameters.p);
}

double ModelPipeSequential::q() { return q_; }
double ModelPipeSequential::dq_dp_in() { return dq_dp_in_; }
double ModelPipeSequential::dq_dp_out() { 
  return dq_dp_out_; }

void ModelPipeSequential::CallFindSequentialQ(
    const Gas &gas_in,
    const Gas &gas_out,
    const PassportPipe &passport,
    const int number_of_segments,
    double *t_out,
    double *q_out) {
  FindSequentialQ(
      // Давление, которое должно получиться в конце.
      gas_out.work_parameters.p, 
      // Рабочие параметры газового потока на входе.
      gas_in.work_parameters.p, 
      gas_in.work_parameters.t,  
      // Состав газа.
      gas_in.composition.density_std_cond, 
      gas_in.composition.co2, 
      gas_in.composition.n2, 
      // Свойства трубы.
      passport.d_inner_, 
      passport.d_outer_, 
      passport.roughness_coeff_, 
      passport.hydraulic_efficiency_coeff_, 
      // Свойства внешней среды (тоже входят в паспорт трубы).
      passport.t_env_, 
      passport.heat_exchange_coeff_, 
      // Длина сегмента и количество сегментов.
      passport.length_/number_of_segments, 
      number_of_segments, 
      // out - параметры, значения t и q на выходе.
      t_out, 
      q_out 
    ); // Конец FindSequentialQ. 
}

double ModelPipeSequential::FindQWithDeltaPIn(double delta_p_in, int segments) {
  Gas real_in = gas_in_;
  Gas real_out = gas_out_;
  if(IsReverse() == true) {
    real_in = gas_out_;
    real_out = gas_in_;
  }
  Gas gas_dp_in = real_in;  
  gas_dp_in.work_parameters.p += delta_p_in;  
  double t_dummy;
  double q_of_dp_in;
  CallFindSequentialQ( // q(Pвх + delta_p_in, Pвых)
    gas_dp_in, real_out, passport_, segments, &( t_dummy ), &q_of_dp_in );
  return q_of_dp_in;
}
double ModelPipeSequential::FindQWithDeltaPOut(double delta_p_out, int segments) {
  Gas real_in = gas_in_;
  Gas real_out = gas_out_;
  if(IsReverse() == true) {
    real_in = gas_out_;
    real_out = gas_in_;
  }  
  Gas gas_dp_out = real_out;
  gas_dp_out.work_parameters.p += delta_p_out;
  double t_dummy;
  double q_of_dp_out;
  CallFindSequentialQ( // q(Pвх, Pвых + delta_p_in)
    real_in, gas_dp_out, passport_, segments,&t_dummy, &q_of_dp_out);
  return q_of_dp_out;
}
// Расчёт трубы. По заданным gas_in, gas_out - найти q, производные.
void ModelPipeSequential::Count() {
	/* В зависимости от направления потока по трубе реально входом может
  являться вход (прямое течение), или выход (реверсивное) - учитываем.*/
  Gas real_in = gas_in_;
  Gas real_out = gas_out_;
  if(IsReverse() == true) {
    real_in = gas_out_;
    real_out = gas_in_;
  }
  /* Температура, которая получится на выхое трубы, т.е. на выходе, если
  течение прямое, или на входе, если течение реверсивное.*/
  double t_res = 0; 
  // Для рассчитанных температур, отбрасывах при расчёте производных.
  double t_dummy = 0; 
  /*Ф-я расчёта q всегда принимает Pвх > Pвых и возвращает q > 0.*/
	/** \todo: как-то разобраться цивилизованно с количеством сегментов
  при расчёте трубы методом последовательного счёта.*/
	int segments = std::max( 1, static_cast<int>(passport_.length_)/10 ); 
  CallFindSequentialQ( // Расчитываем q(Pвх, Pвых)
      real_in, real_out, passport_, segments, 
      &( t_res ), &q_ ); 
  
  /** \todo Разобраться с точностью дифференцирования, понять,
  какие условия останова задать в методе дихотомии для расчёта расхода.
  Сейчас в зависимости от приращения дифференцирования eps получаются
  совершенно разные значения производной, и какого-то определённого предела
  не просматривается. Надо разобраться как соотносятся точности метода и
  шаг дифференцирования. Попробовать центральную производную. 
  Построить графики, чтобы визуально оценить наличие предела.*/
  
  // Рассчитываем производные.
  // Следим, чтобы шаг дифференцирования был не больше (Pвых - Pвх) / 2.
  double eps = std::min(
      0.000005, // Значение по умолчанию - 5 Па.
      (real_in.work_parameters.p - real_out.work_parameters.p)/2 // dP/2.
  ); 

  // Расчитываем производную q по p_вх, p_вых с четвёртым порядоком точности.
  double q_p_in_plus_eps =  FindQWithDeltaPIn( eps, segments); 
  //double q_p_in_plus_2eps = FindQWithDeltaPIn( 2*eps, segments);
  double q_p_in_minus_eps = FindQWithDeltaPIn( -eps, segments);
  //double q_p_in_minus_2eps = FindQWithDeltaPIn(-2*eps, segments);
  /*dq_dp_in_ = (
     q_p_in_minus_2eps  
    -q_p_in_minus_eps*27
    +q_p_in_plus_eps*27 
    -q_p_in_plus_2eps) / (24*eps);*/
  dq_dp_in_ = (q_p_in_plus_eps - q_p_in_minus_eps) / (2*eps);

  double q_p_out_plus_eps =  FindQWithDeltaPOut( eps, segments); 
  //double q_p_out_plus_2eps = FindQWithDeltaPOut( 2*eps, segments);
  double q_p_out_minus_eps = FindQWithDeltaPOut( -eps, segments);
  //double q_p_out_minus_2eps = FindQWithDeltaPOut(-2*eps, segments);
  /*dq_dp_out_ = (
     q_p_out_minus_2eps  
    -q_p_out_minus_eps*27 
    +q_p_out_plus_eps*27 
    -q_p_out_plus_2eps) / (24*eps);*/
  dq_dp_out_ = (q_p_out_plus_eps - q_p_out_minus_eps) / (2*eps);
   
  // Если труба реверсивна - расход отрицательный, 
  // производные - меняются местами и знаком, температура выхода идёт во вход.
  if( IsReverse() ) {
		q_ = -q_;
    std::swap(dq_dp_in_, dq_dp_out_);
    dq_dp_in_ = -dq_dp_in_;
    dq_dp_out_ = -dq_dp_out_;
    // Записываем расчитанную температуру в входной газовый поток.
    gas_in_.work_parameters.t = t_res;
	}
  else { // Прямое течение. Ничего менять не надо, температура идёт на выход.
    gas_out_.work_parameters.t = t_res;
  }
  // Записываем полученный расход в входящий и исходящий газовый поток.
	gas_in_.work_parameters.q = q_;
	gas_out_.work_parameters.q = q_;
}
