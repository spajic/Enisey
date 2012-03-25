#include "model_pipe_sequential.h"

//#include "model_pipe_sequential_functions.cuh" - теперь в functions_pipe.h
#include "functions_pipe.h"

ModelPipeSequential::ModelPipeSequential(const Passport* passport)
{
	passport_ = *(dynamic_cast<const PassportPipe*>(passport));
}

ModelPipeSequential::ModelPipeSequential()
{

}

void ModelPipeSequential::set_gas_in(const Gas* gas)
{
	gas_in_ = *gas;
}

void ModelPipeSequential::set_gas_out(const Gas* gas)
{
	gas_out_ = *gas;
}

void ModelPipeSequential::DetermineDirectionOfFlow()
{
	if(gas_in_.work_parameters.p > gas_out_.work_parameters.p)
	{
		direction_is_forward_ = true;
	}
	else
	{
		direction_is_forward_ = false;
	}
}

float ModelPipeSequential::q()
{
	return q_;
}

// Расчёт трубы. По заданным gas_in, gas_out - найти q.
void ModelPipeSequential::Count()
{
	// В зависимости от направления - решаем, как вызвать функцию расчёта q,
	// но q этой функцией всегда возвращается положительным.
	// Мы же здесь в придаём q отрицательный знак, если направление потока
	// изменено.

	// Определяем направление течения газа.
	DetermineDirectionOfFlow(); 
	
	int number_of_segments = 10; // ToDo: как-то разобраться с этим цивилизованно.

	if(direction_is_forward_ == true)
	{
		FindSequentialQ(
			gas_out_.work_parameters.p, // давление, которое должно получиться в конце
			gas_in_.work_parameters.p, gas_in_.work_parameters.t,  // рабочие параметры газового потока на входе
			gas_in_.composition.density_std_cond, gas_in_.composition.co2, gas_in_.composition.n2, // состав газа
			passport_.d_inner_, passport_.d_outer_, passport_.roughness_coeff_, passport_.hydraulic_efficiency_coeff_, // св-ва трубы
			passport_.t_env_, passport_.heat_exchange_coeff_, // св-ва внешней среды (тоже входят в пасспорт трубы)
			passport_.length_/number_of_segments, number_of_segments, // длина сегмента и кол-во сегментов
			&(gas_out_.work_parameters.t), &q_ ); // out - параметры, значения на выходе )
	}
	else // direction_is_forward == false
	{
		FindSequentialQ(
			gas_in_.work_parameters.p, // давление, которое должно получиться в конце
			gas_out_.work_parameters.p, gas_out_.work_parameters.t,  // рабочие параметры газового потока на входе
			gas_out_.composition.density_std_cond, gas_out_.composition.co2, gas_out_.composition.n2, // состав газа
			passport_.d_inner_, passport_.d_outer_, passport_.roughness_coeff_, passport_.hydraulic_efficiency_coeff_, // св-ва трубы
			passport_.t_env_, passport_.heat_exchange_coeff_, // св-ва внешней среды (тоже входят в пасспорт трубы)
			passport_.length_/number_of_segments, number_of_segments, // длина сегмента и кол-во сегментов
			&(gas_out_.work_parameters.t), &q_ ); // out - параметры, значения на выходе )

			// Направление потока изменено на обратное, поэтому расход возвращаем с минусом.
			q_ = -q_;
	}
	
	gas_in_.work_parameters.q = q_;
	gas_out_.work_parameters.q = q_;
}

/*
//-----------------------------------------------------------------------------
//---------------------------- Наследие Belek ---------------------------------
//-----------------------------------------------------------------------------
// ToDo: Разработать соглашение о написании заголовков
// ToDo: Написать заголовок для этого файла
#include "EdgePipeSequential.h"

// ToDo: Присвоить функциям соответствующие модификаторы static и const
// прочитать об этом у Мейерса в Effective C++
// ToDo: идея об именовании функций - по-разному называть мутаторы и не мут-ры
// тоже посмотреть, какие соглашения на этот счёт бывают, в т.ч. у Мейерса

// Произвести расчёт трубы, то есть:
// Должны быть заданы gas_in и gas_out
// при известных (Pвх, Tвх, Pвх) рассчитать (Q, Tвых)
// рассчитать производные на dQ/dPвх, dQ/dPвых
__host__ __device__
void EdgePipeSequential::Count()
{
	// Проверим, что давление на входе больше, чем давление на выходе
	// если это не так, то газ должен течь в другую сторону
	// ToDo: определиться, что делать в случае Pвх < Pвых
	// Здесь будет использоваться функция типа FindQ - рассчитывающая Q
	// Наверное, стоит начать с неё, чтобы определиться с особенностями
	// её использования для расчёта производных

	// ToDo: определиться, что делать, если в Pвх < Pвых
	// ToDo: определиться, как должна обрабатываться ситуация, когда
	// давление газа задано - в этом случае производная всегда равна нулю
	//if (GasFlowIn.getPisReady() == true)
	//{ 
	//	ProizvIn = 0;
	//}

	// Рассчитываем значение Q для трубы
	float q = ReturnQSequential();
	// Рассчитываем производную dQ/dPвх
	// ToDo: учесть, что производная не может быть отризательна
	// растёт давление на входе, значит растёт и кол-во газа

	// ToDo: сделать чёткое и ясное задание eps
	float eps = 0.001;

	// Производные! Чтобы их считать, нужно задавать приращения - вопрос - где?
	// Сохраню исходные gas_in и gas_out трубы, буду задавать приращения,
	// и возвращатб объекты в исходное состояние.
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