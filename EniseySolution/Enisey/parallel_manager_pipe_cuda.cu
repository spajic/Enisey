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

#include "cuprintf.cu"

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
  cuPrintfRestrict(0, 0);
    cuPrintf("FindQResultCudaKernel-----------------------------------\n");
	int index = threadIdx.x + blockIdx.x * blockDim.x;
  while(index < size)
	{
		// Загружаем данные
		// Состав газа
    cuPrintf("Loading data from memory--------------------------------\n");
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
    cuPrintf("Pipe Passport parameters:-------------------------------\n");
    cuPrintf("length = %f\n", length_);
    cuPrintf("d_in   = %f\n", d_in_out_.x);
    cuPrintf("d_out  = %f\n", d_in_out_.y);
    cuPrintf("hydr   = %f\n", hydr_rough_env_exch_.w);
    cuPrintf("rough  = %f\n", hydr_rough_env_exch_.x);
    cuPrintf("env    = %f\n", hydr_rough_env_exch_.y);
    cuPrintf("exch   = %f\n", hydr_rough_env_exch_.z);
    cuPrintf("Work parameters:---------------------------------------\n");
    cuPrintf("p      = %f\n", p_and_t_.x);
    cuPrintf("p_out  = %f\n", p_target_);
    cuPrintf("t      = %f\n", p_and_t_.y);
    cuPrintf("den_sc = %f\n", den_sc_);
    cuPrintf("co2    = %f\n", co2_);
    cuPrintf("n2     = %f\n", n2_);        
		// Вычисляем базовые свойства газового потока
		double r_sc_ = FindRStandartConditionsCuda(den_sc_); // Совпадает c CPU.
		double t_pc_ = FindTPseudoCriticalCuda(den_sc_, co2_, n2_); // Совпадает.
		double p_pc_ = FindPPseudoCriticalCuda(den_sc_, co2_, n2_); // Совпадает.

		double q_out = 0;
		double p_out = 0;
		double t_out = 0;
		
    int segments = static_cast<int>(length_)/10;
    if(segments < 1) segments = 1;

		FindSequentialQCuda(
        p_target_, // давление, которое должно получиться в конце
        p_and_t_.x, p_and_t_.y,  // рабочие параметры газового потока на входе
        den_sc_, co2_, n2_, // состав газа
        d_in_out_.x, d_in_out_.y, 
        hydr_rough_env_exch_.x, //rough
        hydr_rough_env_exch_.w, //hydr
        hydr_rough_env_exch_.y, //env
        hydr_rough_env_exch_.z, //heat_exch
        length_/segments, segments, // длина сегмента и кол-во сегментов
        &t_out, &q_out); // out - параметры, значения на выходе )

		q_result[index] = q_out;
		
		index += gridDim.x * blockDim.x;
	} // end while (index < size)
}

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
  cudaPrintfInit();
  FindQResultCudaKernel<<<512, 64, 0>>>(
			number_of_pipes,
			den_sc_dev_, co2_dev_, n2_dev_,
			p_in_and_t_in_dev_, p_target_dev_,
			length_dev_,
			d_in_out_dev_,
			hydr_rough_env_exch_dev_,
			q_result_dev_);
  cudaPrintfDisplay(stdout, true);
  cudaPrintfEnd();
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
  length_dev_vec              .resize(number_of_pipes);
  d_in_out_dev_vec            .resize(number_of_pipes);
  hydr_rough_env_exch_dev_vec .resize(number_of_pipes);
  // Рабочие параметры.
  p_in_and_t_in_dev_vec       .resize(number_of_pipes);
  p_target_dev_vec            .resize(number_of_pipes);  
  den_sc_dev_vec              .resize(number_of_pipes);
  co2_dev_vec                 .resize(number_of_pipes);
  n2_dev_vec                  .resize(number_of_pipes);
  // Рассчитанные параметры.
  q_result_dev_vec            .resize(number_of_pipes);

  // Формируем указатели на память device.
  using thrust::raw_pointer_cast;
  den_sc_dev_             = raw_pointer_cast(&den_sc_dev_vec              [0]);
  co2_dev_                = raw_pointer_cast(&co2_dev_vec                 [0]);
  n2_dev_                 = raw_pointer_cast(&n2_dev_vec                  [0]);
  p_in_and_t_in_dev_      = raw_pointer_cast(&p_in_and_t_in_dev_vec       [0]);
  p_target_dev_           = raw_pointer_cast(&p_target_dev_vec            [0]);
  length_dev_             = raw_pointer_cast(&length_dev_vec              [0]);
  d_in_out_dev_           = raw_pointer_cast(&d_in_out_dev_vec            [0]);
  hydr_rough_env_exch_dev_= raw_pointer_cast(&hydr_rough_env_exch_dev_vec [0]);
  q_result_dev_           = raw_pointer_cast(&q_result_dev_vec            [0]);
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
