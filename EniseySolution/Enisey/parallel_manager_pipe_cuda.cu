/** \file parallel_manager_pipe_cuda.cpp
Реализация менеджера параллельного моделирования труб, 
использующего CUDA.*/
#include "parallel_manager_pipe_cuda.cuh"

#include <vector>
#include "calculated_params.h"

// Для векторных типов CUDA double2, double4 и т.д.
#include <vector_types.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "model_pipe_sequential_functions_cuda.cuh"
#include "gas_count_functions_cuda.cuh"

ParallelManagerPipeCUDA::ParallelManagerPipeCUDA() {
  number_of_pipes = 0;
}
void ParallelManagerPipeCUDA::
  TakeUnderControl(std::vector<PassportPipe> const &passports) { 
    number_of_pipes = passports.size();
    AllocateMemoryOnHost();
    SavePassportsAsStructureOfArrays(passports);    
    AllocateMemoryOnDevice();
    SendPassportsToDevice();
}
void ParallelManagerPipeCUDA::
    SetWorkParams(std::vector<WorkParams> const &work_params) {    
  SaveWorkParamsToStructureOfArrays(work_params);
  SendWorkParamsToDevice();
}
void ParallelManagerPipeCUDA::CalculateAll() {
  // Вызов соответствующих CUDA-kernel.
  SendCalculatedParamsToHost();
}
void ParallelManagerPipeCUDA::
    GetCalculatedParams(std::vector<CalculatedParams> *calculated_params){
  calculated_params->erase(
      calculated_params->begin(), calculated_params->end() );
  calculated_params->reserve(number_of_pipes);
  for(auto q = q_result_host.begin(); q != q_result_host.end(); ++q) {
    CalculatedParams cp;
    cp.set_q(*q);
    calculated_params->push_back(cp);
  }
}

void ParallelManagerPipeCUDA::AllocateMemoryOnHost() {
  // Пасспортные параметры.
  length_host               .reserve(number_of_pipes);
  d_in_out_host             .reserve(number_of_pipes);
  hydr_rough_env_exch_host  .reserve(number_of_pipes);
  // Рабочие параметры.
  den_sc_host               .reserve(number_of_pipes);
  co2_host                  .reserve(number_of_pipes);
  n2_host                   .reserve(number_of_pipes);
  p_in_and_t_in_host        .reserve(number_of_pipes);
  p_target_host             .reserve(number_of_pipes);
  // Рассчитанные параметры.
  q_result_host             .reserve(number_of_pipes);
}
void ParallelManagerPipeCUDA::AllocateMemoryOnDevice() {
  // Пасспортные параметры. 
  length_dev_vec              .reserve(number_of_pipes);
  d_in_out_dev_vec            .reserve(number_of_pipes);
  hydr_rough_env_exch_dev_vec .reserve(number_of_pipes);
  // Рабочие параметры.
  p_in_and_t_in_dev_vec       .reserve(number_of_pipes);
  p_target_dev_vec            .reserve(number_of_pipes);  
  den_sc_dev_vec              .reserve(number_of_pipes);
  co2_dev_vec                 .reserve(number_of_pipes);
  n2_dev_vec                  .reserve(number_of_pipes);
  // Рассчитанные параметры.
  q_result_dev_vec            .reserve(number_of_pipes);
}
void ParallelManagerPipeCUDA::
    SavePassportsAsStructureOfArrays(
        std::vector<PassportPipe> const &passports) {
  for(auto p = passports.begin(); p != passports.end(); ++p) {
    double length = p->length_;
    double2 d_in_out = make_double2(p->d_inner_, p->d_outer_);
    double4 hydr_rough_env_exch = 
        make_double4(
            p->hydraulic_efficiency_coeff_,
            p->roughness_coeff_,
            p->t_env_,
            p->heat_exchange_coeff_ );
    length_host.push_back(length);
    d_in_out_host.push_back(d_in_out);
    hydr_rough_env_exch_host.push_back(hydr_rough_env_exch);
  }
}
void ParallelManagerPipeCUDA::
    SaveWorkParamsToStructureOfArrays(
        std::vector<WorkParams> const &work_params){
  int index_in_vector = 0;
  for(auto wp = work_params.begin(); wp != work_params.end(); ++wp) {
    den_sc_host       [index_in_vector] = wp->den_sc_in();
    co2_host          [index_in_vector] = wp->co2_in();
    n2_host           [index_in_vector] = wp->n2_in();
    p_in_and_t_in_host[index_in_vector] = 
                            make_double2( wp->p_in(), 
                                          wp->t_in() );
    p_target_host     [index_in_vector] = wp->p_out();
    ++index_in_vector;
  }
}

void ParallelManagerPipeCUDA::SendPassportsToDevice() {
  // Магия thrust. Присваивание dev_vec = host_vec отправляет вектор на Device.
  length_dev_vec              = length_host;
  d_in_out_dev_vec            = d_in_out_host;
  hydr_rough_env_exch_dev_vec = hydr_rough_env_exch_host;
}
void ParallelManagerPipeCUDA::SendWorkParamsToDevice() {
  den_sc_dev_vec        = den_sc_host;
  co2_dev_vec           = co2_host;
  n2_dev_vec            = n2_host;
  p_in_and_t_in_dev_vec = p_in_and_t_in_host;
  p_target_dev_vec      = p_target_host;
}
void ParallelManagerPipeCUDA::SendCalculatedParamsToHost() {
  q_result_host = q_result_dev_vec;
}

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
		double r_sc_ = FindRStandartConditionsCuda(den_sc_); // Совпадает c CPU.
		double t_pc_ = FindTPseudoCriticalCuda(den_sc_, co2_, n2_); // Совпадает.
		double p_pc_ = FindPPseudoCriticalCuda(den_sc_, co2_, n2_); // Совпадает.

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