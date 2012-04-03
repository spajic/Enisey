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

float ModelPipeSequential::q() {
	return q_;
}

// Расчёт трубы. По заданным gas_in, gas_out - найти q.
void ModelPipeSequential::Count() {
	/* В зависимости от направления потока по трубе реально входом может
  являться вход (прямое течение), или выход (реверсивное) - учитываем.*/
  Gas real_in = gas_in_;
  Gas real_out = gas_out_;
  if(IsReverse() == true) {
    real_in = gas_out_;
    real_out = gas_in_;
  }
  /*Ф-я расчёта q всегда принимает Pвх > Pвых и возвращает q > 0.*/
	/// \todo: как-то разобраться с этим цивилизованно
	int number_of_segments = 10; 

	FindSequentialQ(
      // Давление, которое должно получиться в конце.
	    real_out.work_parameters.p, 
      // Рабочие параметры газового потока на входе.
			real_in.work_parameters.p, 
      real_in.work_parameters.t,  
      // Состав газа.
			real_in.composition.density_std_cond, 
      real_in.composition.co2, 
      real_in.composition.n2, 
      // Свойства трубы.
			passport_.d_inner_, 
      passport_.d_outer_, 
      passport_.roughness_coeff_, 
      passport_.hydraulic_efficiency_coeff_, 
      // Свойства внешней среды (тоже входят в паспорт трубы).
			passport_.t_env_, 
      passport_.heat_exchange_coeff_, 
      // Длина сегмента и количество сегментов.
			passport_.length_/number_of_segments, number_of_segments, 
      // out - параметры, значения t и q на выходе.
			&(real_out.work_parameters.t), 
      &q_ 
  ); // Конец FindSequentialQ.
  // Если труба реверсивна - расход отрицательный.
  if( IsReverse() ) {
		q_ = -q_;
	}
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