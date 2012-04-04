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
void ModelPipeSequential::set_gas_out(const Gas* gas) {
	gas_out_ = *gas;
}

inline
bool ModelPipeSequential::IsReverse() {
  return (gas_in_.work_parameters.p < gas_out_.work_parameters.p);
}

float ModelPipeSequential::q() { return q_; }
float ModelPipeSequential::dq_dp_in() { return dq_dp_in_; }
float ModelPipeSequential::dq_dp_out() { return dq_dp_out_; }

void ModelPipeSequential::CallFindSequentialQ(
    const Gas &gas_in,
    const Gas &gas_out,
    const PassportPipe &passport,
    const int number_of_segments,
    float *t_out,
    float *q_out) {
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
  float t_res = 0; 
  // Для рассчитанных температур, отбрасывах при расчёте производных.
  float t_dummy = 0; 
  /*Ф-я расчёта q всегда принимает Pвх > Pвых и возвращает q > 0.*/
	/// \todo: как-то разобраться с этим цивилизованно
	int segments = 10; 
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
  float eps = 0.0001; 

  // Расчитываем производную q по p_вх.
  float q_p_in_plus_eps(0.0); // q(p_вх + eps, p_вых).
  Gas gas_dq_dp_in = real_in;
  gas_dq_dp_in.work_parameters.p += eps;
  CallFindSequentialQ( // q(Pвх + eps, Pвых)
      gas_dq_dp_in, real_out, passport_, segments,
      &( t_dummy ), &q_p_in_plus_eps );
  dq_dp_in_ = (q_p_in_plus_eps - q_) / eps;

  // Рассчитываем производную q по p_вых.
  float q_p_out_plus_eps(0.0); // q(p_вх, p_вых + eps).
  Gas gas_dq_dp_out = real_out;
  gas_dq_dp_out.work_parameters.p += eps;
  CallFindSequentialQ( // q(Pвх, Pвых + eps)
      real_in, gas_dq_dp_out, passport_, segments,
      &( t_dummy ), &q_p_out_plus_eps );
  dq_dp_out_ = (q_p_out_plus_eps - q_) / eps;
   
  // Если труба реверсивна - расход отрицательный, 
  // производные - меняются местами и знаком, температура выхода идёт во вход.
  if( IsReverse() ) {
		q_ = -q_;
    std::swap(dq_dp_in_, dq_dp_out_);
    dq_dp_in_ = -dq_dp_in_;
    dq_dp_out_ = -dq_dp_out_;
    // Записываем расчитанную температуру в входной газовый поток.
    gas_in_.work_parameters.t = t_res;
    //float swap = dq_dp_out_;
    //dq_dp_out_ = -dq_dp_in_;
    //dq_dp_in_ = -swap;
	}
  else { // Прямое течение. Ничего менять не надо, температура идёт на выход.
    gas_out_.work_parameters.t = t_res;
  }
  // Записываем полученный расход в входящий и исходящий газовый поток.
	gas_in_.work_parameters.q = q_;
	gas_out_.work_parameters.q = q_;
}

/*
//-----------------------------------------------------------------------------
//---------------------------- Наследие Belek ---------------------------------
//-----------------------------------------------------------------------------

#include "EdgePipeSequential.h"

// ToDo: Присвоить функциям соответствующие модификаторы static и const
// прочитать об этом у Мейерса в Effective C++
// ToDo: идея об именовании функций - по-разному называть мутаторы и не мут-ры
// тоже посмотреть, какие соглашения на этот счёт бывают, в т.ч. у Мейерса

// рассчитать производные на dQ/dPвх, dQ/dPвых
__host__ __device__
void EdgePipeSequential::Count()
{
	// ToDo: определиться, как должна обрабатываться ситуация, когда
	// давление газа задано - в этом случае производная всегда равна нулю
	//if (GasFlowIn.getPisReady() == true)
	//{ 
	//	ProizvIn = 0;
	//}

	// Рассчитываем значение Q для трубы
	float q = ReturnQSequential();
	// Рассчитываем производную dQ/dPвх
	// ToDo: учесть, что производная не может быть отрицательна
	// растёт давление на входе, значит растёт и кол-во газа

	// ToDo: сделать чёткое и ясное задание eps
	float eps = 0.001;

	// Производные! Чтобы их считать, нужно задавать приращения - вопрос - где?
	// Сохраню исходные gas_in и gas_out трубы, буду задавать приращения,
	// и возвращать объекты в исходное состояние.
	ReducedGasComposition gas_in_prototype = gas_in_;
	ReducedGasComposition gas_out_prototype = gas_out_;

	// Производная dQ/dPin
	// Производная в правой точке
	gas_in_.SetPWorkAndTWork(gas_in_prototype.p_work() + eps, 
		gas_in_prototype.t_work());
	float right = ReturnQSequential();
	// Производная в левой точке
	gas_in_.SetPWorkAndTWork(gas_in_prototype.p_work() - eps, 
		gas_in_prototype.t_work());
	float left = ReturnQSequential();
	// Находим центральную производную dQ/dPin
	derivative_of_q_by_p_in_ = (right-left)/(2*eps);
	// Возвращаем состояние газового объекта
	gas_in_ = gas_in_prototype;

	// Рассчитываем производную dQ/dPвых
	// ToDo: учесть, что производная не может быть больше нуля
	// увеличиваем давление на выходе, кол-во газа падает
	// Производная в правой точке
	gas_out_.SetPWorkAndTWork(gas_out_prototype.p_work() + eps,
		gas_out_prototype.t_work());
	right = ReturnQSequential();
	// Производная в левой точке
	gas_out_.SetPWorkAndTWork(gas_out_prototype.p_work() - eps, 
		gas_out_prototype.t_work());
	left = ReturnQSequential();
	// Находим центральную производную dQ/dPвых
	derivative_of_q_by_p_out_ = (right-left)/(2*eps);
	// Возвращаем состояние газового объекта
	gas_out_ = gas_out_prototype;
}
*/