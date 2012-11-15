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

#ifdef CUPRINTF
  #include "cuprintf.cu"
#endif
__global__ 
void FindQResultCudaKernel(
	  int size,
	  double* den_sc, double* co2, double* n2, 
	  double2* p_and_t, double* p_target,
	  double* length,
	  double2* d_in_out,
	  double4* hydr_rough_env_exch,
	  double* q_result,
    double* t_result,
    double* dq_dp_in_result,
    double* dq_dp_out_result) {
  #ifdef CUPRINTF
    cuPrintfRestrict(0, 0);
    cuPrintf("FindQResultCudaKernel------------------------------------\n");
  #endif
	int index = threadIdx.x + blockIdx.x * blockDim.x;
  while(index < size)	{
		// Загружаем данные
		// Состав газа
    #ifdef CUPRINTF
      cuPrintf("Loading data from memory---------------------------------\n");
    #endif
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
    #ifdef CUPRINTF
      cuPrintf("Pipe Passport parameters:--------------------------------\n");
      cuPrintf("length = %f\n", length_);
      cuPrintf("d_in   = %f\n", d_in_out_.x);
      cuPrintf("d_out  = %f\n", d_in_out_.y);
      cuPrintf("hydr   = %f\n", hydr_rough_env_exch_.x);
      cuPrintf("rough  = %f\n", hydr_rough_env_exch_.y);
      cuPrintf("env    = %f\n", hydr_rough_env_exch_.z);
      cuPrintf("exch   = %f\n", hydr_rough_env_exch_.w);    
      cuPrintf("\nWork parameters:---------------------------------------\n");
      cuPrintf("p      = %f\n", p_and_t_.x);
      cuPrintf("p_out  = %f\n", p_target_);
      cuPrintf("t      = %f\n", p_and_t_.y);
      cuPrintf("den_sc = %f\n", den_sc_);
      cuPrintf("co2    = %f\n", co2_);
      cuPrintf("n2     = %f\n", n2_);        
    #endif
		// Вычисляем базовые свойства газового потока    
		double r_sc_ = FindRStandartConditionsCuda(den_sc_); // Совпадает c CPU.
		double t_pc_ = FindTPseudoCriticalCuda(den_sc_, co2_, n2_); // Совпадает.
		double p_pc_ = FindPPseudoCriticalCuda(den_sc_, co2_, n2_); // Совпадает.
    #ifdef CUPRINTF
      cuPrintf("\nCalculating base gas props:----------------------------\n");
      cuPrintf("r_sc = %f\n", r_sc_);
      cuPrintf("t_pc = %f\n", t_pc_);
      cuPrintf("p_pc = %f\n", p_pc_);
    #endif

		double q_out = 0;
		double p_out = 0;
		double t_out = 0;
		
    int segments = static_cast<int>(length_)/10;
    if(segments < 1) { segments = 1; }
    #ifdef CUPRINTF
      cuPrintf("\nCalculating # of segments:-----------------------------\n");
      cuPrintf("segments = %d\n", segments);
    #endif

		FindSequentialQCuda(
        p_target_, // давление, которое должно получиться в конце
        p_and_t_.x, p_and_t_.y,  // рабочие параметры газового потока на входе
        den_sc_, co2_, n2_, // состав газа
        d_in_out_.x, d_in_out_.y, 
        hydr_rough_env_exch_.y, //rough
        hydr_rough_env_exch_.x, //hydr
        hydr_rough_env_exch_.z, //env
        hydr_rough_env_exch_.w, //heat_exch
        length_/segments, segments, // длина сегмента и кол-во сегментов
        &t_out, &q_out); // out - параметры, значения на выходе )

		q_result[index] = q_out;
    t_result[index] = t_out;
    // Рассчитываем производные--------------------------------------------
    // Следим, чтобы шаг дифференцирования был не больше (Pвых - Pвх) / 2.
    double eps = (p_and_t_.x - p_target_)/2; // dP/2.
    if(eps > 0.000005) { eps = 0.000005; } // Значение по умолчанию - 5 Па.  
    double t_dummy; // t, возвращаемое при расчёте производных не нужно.
    double q_p_in_plus_eps; 
    FindSequentialQCuda(
      p_target_ , // давление, которое должно получиться в конце
      p_and_t_.x + eps, p_and_t_.y,  // рабочие параметры газового потока на входе
      den_sc_, co2_, n2_, // состав газа
      d_in_out_.x, d_in_out_.y, 
      hydr_rough_env_exch_.y, //rough
      hydr_rough_env_exch_.x, //hydr
      hydr_rough_env_exch_.z, //env
      hydr_rough_env_exch_.w, //heat_exch
      length_/segments, segments, // длина сегмента и кол-во сегментов
      &t_dummy, &q_p_in_plus_eps); // out - параметры, значения на выходе )
    double q_p_in_minus_eps; 
    FindSequentialQCuda(
      p_target_, // давление, которое должно получиться в конце
      p_and_t_.x - eps, p_and_t_.y,  // рабочие параметры газового потока на входе
      den_sc_, co2_, n2_, // состав газа
      d_in_out_.x, d_in_out_.y, 
      hydr_rough_env_exch_.y, //rough
      hydr_rough_env_exch_.x, //hydr
      hydr_rough_env_exch_.z, //env
      hydr_rough_env_exch_.w, //heat_exch
      length_/segments, segments, // длина сегмента и кол-во сегментов
      &t_dummy, &q_p_in_minus_eps); // out - параметры, значения на выходе )
    dq_dp_in_result[index] = (q_p_in_plus_eps - q_p_in_minus_eps) / (2*eps);

    double q_p_out_plus_eps;
    FindSequentialQCuda(
      p_target_ + eps, // давление, которое должно получиться в конце
      p_and_t_.x, p_and_t_.y,  // рабочие параметры газового потока на входе
      den_sc_, co2_, n2_, // состав газа
      d_in_out_.x, d_in_out_.y, 
      hydr_rough_env_exch_.y, //rough
      hydr_rough_env_exch_.x, //hydr
      hydr_rough_env_exch_.z, //env
      hydr_rough_env_exch_.w, //heat_exch
      length_/segments, segments, // длина сегмента и кол-во сегментов
      &t_dummy, &q_p_out_plus_eps); // out - параметры, значения на выходе )
    double q_p_out_minus_eps;       
    FindSequentialQCuda(
      p_target_ - eps, // давление, которое должно получиться в конце
      p_and_t_.x, p_and_t_.y,  // рабочие параметры газового потока на входе
      den_sc_, co2_, n2_, // состав газа
      d_in_out_.x, d_in_out_.y, 
      hydr_rough_env_exch_.y, //rough
      hydr_rough_env_exch_.x, //hydr
      hydr_rough_env_exch_.z, //env
      hydr_rough_env_exch_.w, //heat_exch
      length_/segments, segments, // длина сегмента и кол-во сегментов
      &t_dummy, &q_p_out_minus_eps); // out - параметры, значения на выходе )
    dq_dp_out_result[index] = (q_p_out_plus_eps - q_p_out_minus_eps) / (2*eps);
    
    #ifdef CUPRINTF
      cuPrintf("\nCalculating results:-----------------------------------\n");
      cuPrintf("q                   = %f\n", q_result[index]);
      cuPrintf("t_out               = %f\n", t_result[index]);
      cuPrintf("dq_dp_in            = %f\n", dq_dp_in_result[index]);
      cuPrintf("--q_p_in_plus_eps     = %f\n", q_p_in_plus_eps);
      cuPrintf("--q_p_in_minus_eps    = %f\n", q_p_in_minus_eps);
      cuPrintf("dq_dp_out           = %f\n", dq_dp_out_result[index]);
      cuPrintf("--q_p_out_plus_eps    = %f\n", q_p_out_plus_eps);
      cuPrintf("--q_p_out_minus_eps   = %f\n", q_p_out_minus_eps);
      cuPrintf("eps                 = %f\n", eps);
    #endif
		
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
  #ifdef CUPRINTF
    cudaPrintfInit();
  #endif
  FindQResultCudaKernel<<<512, 64, 0>>>(
			number_of_pipes,
			den_sc_dev_, co2_dev_, n2_dev_,
			p_in_and_t_in_dev_, p_target_dev_,
			length_dev_,
			d_in_out_dev_,
			hydr_rough_env_exch_dev_,
			q_result_dev_,
      t_result_dev_,
      dq_dp_in_result_dev_,
      dq_dp_out_result_dev_);
  #ifdef CUPRINTF
    cudaPrintfDisplay(stdout, true);
    cudaPrintfEnd();
  #endif
  SendCalculatedParamsToHost();
}
void ParallelManagerPipeCUDA::
    GetCalculatedParams(std::vector<CalculatedParams> *calculated_params){
  calculated_params->erase(
      calculated_params->begin(), calculated_params->end() );
  calculated_params->reserve(number_of_pipes);

  auto t = t_result_host.begin();
  auto dq_dp_in = dq_dp_in_result_host.begin();
  auto dq_dp_out = dq_dp_out_result_host.begin();
  for(auto q = q_result_host.begin(); q != q_result_host.end(); ++q) {
    CalculatedParams cp;
    cp.set_q(*q);
    cp.set_dq_dp_in(*dq_dp_in);
    cp.set_dq_dp_out(*dq_dp_out);
    cp.set_t_out(*t);
    calculated_params->push_back(cp);    
    ++dq_dp_in;
    ++dq_dp_out;
    ++t;
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
  t_result_host             .reserve(number_of_pipes);
  dq_dp_in_result_host      .reserve(number_of_pipes);
  dq_dp_out_result_host     .reserve(number_of_pipes);
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
  t_result_dev_vec            .resize(number_of_pipes);
  dq_dp_in_result_dev_vec     .resize(number_of_pipes);
  dq_dp_out_result_dev_vec    .resize(number_of_pipes);

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
  t_result_dev_           = raw_pointer_cast(&t_result_dev_vec            [0]);
  dq_dp_in_result_dev_    = raw_pointer_cast(&dq_dp_in_result_dev_vec     [0]);
  dq_dp_out_result_dev_   = raw_pointer_cast(&dq_dp_out_result_dev_vec    [0]);
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
  for(auto wp = work_params.begin(); wp != work_params.end(); ++wp) {
    den_sc_host       .push_back( wp->den_sc_in() );
    co2_host          .push_back( wp->co2_in()    );
    n2_host           .push_back( wp->n2_in()     );
    p_in_and_t_in_host.push_back( 
                    make_double2( wp->p_in(), 
                                  wp->t_in() )     );
    p_target_host      .push_back( wp->p_out()    );
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
  q_result_host         = q_result_dev_vec;
  t_result_host         = t_result_dev_vec;
  dq_dp_in_result_host  = dq_dp_in_result_dev_vec;
  dq_dp_out_result_host = dq_dp_out_result_dev_vec;
}
