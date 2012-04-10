#include "manager_edge_model_pipe_sequential_cuda.cuh"

#include "cutil_inline.h"

#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

#include "gas.h"
#include "edge.h"
#include "passport.h"
#include "passport_pipe.h"

#include "model_pipe_sequential_functions_cuda.cuh"
#include "edge_model_pipe_sequential_cuda.cuh"

__global__ 
void FindQResultCudaKernel(
	int size,
	double* den_sc, double* co2, double* n2, 
	double2* p_and_t, double* p_target,
	double* length,
	double2* d_in_out,
	double4* hydr_rough_env_exch,
	double* q_result
)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
  while(index < size)
	{
		// Загружаем данные
		// Состав газа
		double den_sc_ = den_sc[index];
		double co2_ = co2[index];
		double n2_ = n2[index];
		// Давление и температура на входе
		double2 p_and_t_ = p_and_t[index];
		// Пасспотные параметры трубы
		double length_ = length[index];
		double2 d_in_out_ = d_in_out[index];
		double4 hydr_rough_env_exch_ = hydr_rough_env_exch[index];
		double p_target_ = p_target[index];
		
		// Вычисляем базовые свойства газового потока
		double r_sc_ = FindRStandartConditionsCuda(den_sc_);
		double t_pc_ = FindTPseudoCriticalCuda(den_sc_, co2_, n2_);
		double p_pc_ = FindPPseudoCriticalCuda(den_sc_, co2_, n2_);

		double q_out = 0;
		double p_out = 0;
		double t_out = 0;
		
		FindSequentialQCudaRefactored(
			 p_target_,
			 p_and_t_.x,  p_and_t_.y,  // рабочие параметры газового потока на входе
			 p_pc_,  t_pc_,  r_sc_,  den_sc_,
			 d_in_out_.x,  d_in_out_.y,  hydr_rough_env_exch_.y,  hydr_rough_env_exch_.x, // св-ва трубы
			 hydr_rough_env_exch_.z,  hydr_rough_env_exch_.w, // св-ва внешней среды (тоже входят в пасспорт трубы)
			 length_/10, 10, // длина сегмента и кол-во сегментов
			 &p_out, &t_out,
			 &q_out); // out - параметры, значения на выходе 

		q_result[index] = q_out;
		
		index += gridDim.x * blockDim.x;
	} // end while (index < size)
}

ManagerEdgeModelPipeSequentialCuda::ManagerEdgeModelPipeSequentialCuda()
{
	// 1. Получаем кол-во доступных GPU в системе
	cutilSafeCall(cudaGetDeviceCount(&gpu_count_));
	if(gpu_count_ > kMaxGpuCount_)
	{
		gpu_count_ = kMaxGpuCount_;
	}
	
	max_index_ = 0;
	finish_adding_edges_ = false;

	// Резервируем место в векторах.
	edges_.resize(max_count_of_edges);
	
	// ToDo : при нечётном кол-ве GPU >= 3, надо будет подумать - как корректно делить данные
	// (есть в примере из SDK - SimpleMultiGPU)
	for(int i = 0; i < gpu_count_; i++)
	{
		cutilSafeCall( cudaSetDevice(i) );
		cutilSafeCall( cudaStreamCreate(&(thread_data_[i].stream)) );

		cudaMalloc((void**)&(thread_data_[i].length_dev_),				max_count_of_edges * sizeof(double) / gpu_count_);
		cudaMalloc((void**)&thread_data_[i].d_in_out_dev_,				max_count_of_edges * sizeof(double2) / gpu_count_);
		cudaMalloc((void**)&thread_data_[i].hydr_rough_env_exch_dev_,	max_count_of_edges * sizeof(double4) / gpu_count_);
		cudaMalloc((void**)&thread_data_[i].p_in_and_t_in_dev_,			max_count_of_edges * sizeof(double2) / gpu_count_);
		cudaMalloc((void**)&thread_data_[i].p_target_dev_,				max_count_of_edges * sizeof(double) / gpu_count_);
		cudaMalloc((void**)&thread_data_[i].q_result_dev_,				max_count_of_edges * sizeof(double) / gpu_count_);
		cudaMalloc((void**)&thread_data_[i].den_sc_dev_,				max_count_of_edges * sizeof(double) / gpu_count_);
		cudaMalloc((void**)&thread_data_[i].co2_dev_,					max_count_of_edges * sizeof(double) / gpu_count_);
		cudaMalloc((void**)&thread_data_[i].n2_dev_,					max_count_of_edges * sizeof(double) / gpu_count_);

		// Пасспортные параметры
		cutilSafeCall(cudaMallocHost((void**)&(thread_data_[i].length_), max_count_of_edges * sizeof(double) / gpu_count_) );
		cutilSafeCall(cudaMallocHost((void**)&thread_data_[i].d_in_out_, max_count_of_edges * sizeof(double2) / gpu_count_) );
		cutilSafeCall(cudaMallocHost((void**)&thread_data_[i].hydr_rough_env_exch_, max_count_of_edges * sizeof(double4) / gpu_count_) );
		


		// Рабочие параметры
		cutilSafeCall(cudaMallocHost((void**)&(thread_data_[i].p_in_and_t_in_), max_count_of_edges * sizeof(double2) / gpu_count_) );
		cutilSafeCall(cudaMallocHost((void**)&thread_data_[i].p_target_, max_count_of_edges * sizeof(double) / gpu_count_) );
		cutilSafeCall(cudaMallocHost((void**)&thread_data_[i].q_result_, max_count_of_edges * sizeof(double) / gpu_count_) );
		cutilSafeCall(cudaMallocHost((void**)&thread_data_[i].den_sc_, max_count_of_edges * sizeof(double) / gpu_count_) );
		cutilSafeCall(cudaMallocHost((void**)&thread_data_[i].co2_, max_count_of_edges * sizeof(double) / gpu_count_) );
		cutilSafeCall(cudaMallocHost((void**)&thread_data_[i].n2_, max_count_of_edges * sizeof(double) / gpu_count_) );
	}
	
}

ManagerEdgeModelPipeSequentialCuda::~ManagerEdgeModelPipeSequentialCuda()
{
	for(int i = 0; i < gpu_count_; i++)
	{
		cutilSafeCall( cudaSetDevice(i) );

		//Wait for all operations to finish
        cudaStreamSynchronize(thread_data_[i].stream);

		cudaFree(thread_data_[i].length_dev_);
		cudaFree(thread_data_[i].d_in_out_dev_);
		cudaFree(thread_data_[i].hydr_rough_env_exch_dev_);
		cudaFree(thread_data_[i].p_in_and_t_in_dev_);
		cudaFree(thread_data_[i].p_target_dev_);
		cudaFree(thread_data_[i].q_result_dev_);
		cudaFree(thread_data_[i].den_sc_dev_);
		cudaFree(thread_data_[i].co2_dev_);
		cudaFree(thread_data_[i].n2_dev_);

		// Пасспортные параметры
		cutilSafeCall(cudaFreeHost((void**)&thread_data_[i].length_) );
		cutilSafeCall(cudaFreeHost((void**)&thread_data_[i].d_in_out_) );
		cutilSafeCall(cudaFreeHost((void**)&thread_data_[i].hydr_rough_env_exch_) );

		// Рабочие параметры
		cutilSafeCall(cudaFreeHost((void**)&thread_data_[i].p_in_and_t_in_) );
		cutilSafeCall(cudaFreeHost((void**)&thread_data_[i].p_target_) );
		cutilSafeCall(cudaFreeHost((void**)&thread_data_[i].q_result_) );
		cutilSafeCall(cudaFreeHost((void**)&thread_data_[i].den_sc_) );
		cutilSafeCall(cudaFreeHost((void**)&thread_data_[i].co2_) );
		cutilSafeCall(cudaFreeHost((void**)&thread_data_[i].n2_) );

		cutilSafeCall( cudaStreamDestroy(thread_data_[i].stream) );
	}
}


Edge* ManagerEdgeModelPipeSequentialCuda::CreateEdge(const Passport* passport)
{
	PassportPipe pass = *(dynamic_cast<const PassportPipe*>(passport));
	
	int i = 0;
	int index = 0;
	if(max_index_ < max_count_of_edges / gpu_count_)
	{
		i = 0;
		index = max_index_;
	}
	else
	{
		i = 1;
		index = max_index_ - (max_count_of_edges / gpu_count_);
	}
	
	//cutilSafeCall( cudaSetDevice(i) );
	thread_data_[i].length_[index] = pass.length_;
	
	double2 d_in_out_temp;
	d_in_out_temp.x = pass.d_inner_;
	d_in_out_temp.y =  pass.d_outer_;
	thread_data_[i].d_in_out_[index] = d_in_out_temp;
	
	double4 hydr_rough_env_exch_temp;
	hydr_rough_env_exch_temp.x = pass.hydraulic_efficiency_coeff_;
	hydr_rough_env_exch_temp.y = pass.roughness_coeff_;
	hydr_rough_env_exch_temp.z = pass.t_env_;
	hydr_rough_env_exch_temp.w = pass.heat_exchange_coeff_;
	thread_data_[i].hydr_rough_env_exch_[index] = hydr_rough_env_exch_temp;

	EdgeModelPipeSequentialCuda edge(max_index_, this);
	
	edges_[max_index_] = edge;
	++max_index_;
	return &(edges_[max_index_ - 1]);
}

void ManagerEdgeModelPipeSequentialCuda::set_gas_in(const Gas* gas, int index)
{
	int i = 0;
	if(index < max_count_of_edges / gpu_count_)
	{
		i = 0;
	}
	else
	{
		i = 1;
		index -= (max_count_of_edges / gpu_count_);
	}

	// ToDo: Тут и в set_gas_out надо учесть направление потока.
	//cutilSafeCall( cudaSetDevice(i) );

	thread_data_[i].den_sc_[index] = gas->composition.density_std_cond;
	thread_data_[i].co2_[index] = gas->composition.co2;
	thread_data_[i].n2_[index] = gas->composition.n2;

	double2 p_in_and_t_in_temp;
	p_in_and_t_in_temp.x = gas->work_parameters.p;
	p_in_and_t_in_temp.y = gas->work_parameters.t;

	thread_data_[i].p_in_and_t_in_[index] = p_in_and_t_in_temp;
}

void ManagerEdgeModelPipeSequentialCuda::set_gas_out(const Gas* gas, int index)
{
	int i = 0;
	if(index < max_count_of_edges / gpu_count_)
	{
		i = 0;
	}
	else
	{
		i = 1;
		index -= (max_count_of_edges / gpu_count_);
	}

	// ToDo: Тут и в set_gas_out надо учесть направление потока.
	//cutilSafeCall( cudaSetDevice(i) );
	thread_data_[i].p_target_[index] = gas->work_parameters.p;
}

void ManagerEdgeModelPipeSequentialCuda::FinishAddingEdges()
{
	// По завершении добавления рёбер - у нас должна быть собрана
	// вся пасспортная информация о трубах - и её можно отправлять на GPU
	// ToDo: нужно включить в структуру thread_data размер памяти, который обрабатывает каждый поток.
	// причём этот кусок определять аккуратно, с учётом нечётного количества GPU.
	for(int i = 0; i < gpu_count_ ; i++)
	{	
		cutilSafeCall( cudaSetDevice(i) );

		cutilSafeCall(cudaMemcpyAsync(thread_data_[i].length_dev_,				thread_data_[i].length_,				(sizeof(double) * max_count_of_edges) / gpu_count_, cudaMemcpyHostToDevice, thread_data_[i].stream) );
		cutilSafeCall(cudaMemcpyAsync(thread_data_[i].d_in_out_dev_,		thread_data_[i].d_in_out_,				sizeof(double2) * max_count_of_edges / gpu_count_, cudaMemcpyHostToDevice, thread_data_[i].stream) );
		cutilSafeCall(cudaMemcpyAsync(thread_data_[i].hydr_rough_env_exch_dev_,thread_data_[i].hydr_rough_env_exch_,sizeof(double4) * max_count_of_edges / gpu_count_, cudaMemcpyHostToDevice, thread_data_[i].stream) );
	}
}

void ManagerEdgeModelPipeSequentialCuda::CountAll()
{
	if(finish_adding_edges_ == false)
	{
		FinishAddingEdges();
		finish_adding_edges_ = true;
	}

	// Скопировать собранные рабочие параметры на device, пасспортные уже должны быть отправлены
	// туда функцией FinishAddingEdges.
	// (Пасспортные данные каждый раз копировать не надо! Их достаточно скопировать всего один раз)
	for(int i = 0; i < gpu_count_; i++)
	{
		cutilSafeCall( cudaSetDevice(i) );

		cutilSafeCall(cudaMemcpyAsync(thread_data_[i].p_in_and_t_in_dev_, thread_data_[i].p_in_and_t_in_,	sizeof(double2) * max_count_of_edges / gpu_count_, cudaMemcpyHostToDevice, thread_data_[i].stream) );
		cutilSafeCall(cudaMemcpyAsync(thread_data_[i].p_target_dev_, thread_data_[i].p_target_,			sizeof(double) * max_count_of_edges / gpu_count_, cudaMemcpyHostToDevice, thread_data_[i].stream) );
		cutilSafeCall(cudaMemcpyAsync(thread_data_[i].den_sc_dev_, thread_data_[i].den_sc_,				sizeof(double) *max_count_of_edges / gpu_count_, cudaMemcpyHostToDevice, thread_data_[i].stream) );
		cutilSafeCall(cudaMemcpyAsync(thread_data_[i].co2_dev_, thread_data_[i].co2_,						sizeof(double) * max_count_of_edges / gpu_count_, cudaMemcpyHostToDevice, thread_data_[i].stream) );
		cutilSafeCall(cudaMemcpyAsync(thread_data_[i].n2_dev_, thread_data_[i].n2_,						sizeof(double) * max_count_of_edges / gpu_count_, cudaMemcpyHostToDevice, thread_data_[i].stream) );

		// выполняем расчёт на device
		FindQResultCudaKernel<<<512, 64, 0, thread_data_[i].stream>>>(
			max_index_ / gpu_count_,
			thread_data_[i].den_sc_dev_, thread_data_[i].co2_dev_, thread_data_[i].n2_dev_,
			thread_data_[i].p_in_and_t_in_dev_, thread_data_[i].p_target_dev_,
			thread_data_[i].length_dev_,
			thread_data_[i].d_in_out_dev_,
			thread_data_[i].hydr_rough_env_exch_dev_,
			thread_data_[i].q_result_dev_);

		// копируем рассчитанное q обратно на host
		cutilSafeCall(cudaMemcpyAsync(thread_data_[i].q_result_, thread_data_[i].q_result_dev_, sizeof(double) * max_count_of_edges / gpu_count_, cudaMemcpyDeviceToHost, thread_data_[i].stream) );
	}

	for(int i = 0; i < gpu_count_; i++)
	{
		cutilSafeCall( cudaSetDevice(i) );
		cudaStreamSynchronize(thread_data_[i].stream);
	}
}

double ManagerEdgeModelPipeSequentialCuda::q(int index)
{
	int i = 0;
	if(index < max_count_of_edges / gpu_count_)
	{
		i = 0;
	}
	else
	{
		i = 1;
		index -= (max_count_of_edges / gpu_count_);
	}

	return thread_data_[i].q_result_[index];
}