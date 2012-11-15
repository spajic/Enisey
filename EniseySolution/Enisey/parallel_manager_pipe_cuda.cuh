/** \file parallel_manager_pipe_cuda.h
Менеджер параллельного моделирования труб, использующий CUDA.*/
#pragma once
#include "parallel_manager_pipe_i.h"

#include <vector>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using thrust::host_vector;
using thrust::device_vector;

class ParallelManagerPipeCUDA : public ParallelManagerPipeI {
public:
  ParallelManagerPipeCUDA();
  virtual void TakeUnderControl(std::vector<PassportPipe> const &passports);
  virtual void SetWorkParams(std::vector<WorkParams> const &work_params);
  virtual void CalculateAll();
  virtual void GetCalculatedParams(std::vector<CalculatedParams> *calculated_params);
private:  
  void AllocateMemoryOnHost();
  void AllocateMemoryOnDevice();
  void SavePassportsAsStructureOfArrays(
      std::vector<PassportPipe> const &passports);
  void SaveWorkParamsToStructureOfArrays(
      std::vector<WorkParams> const &work_params);
  void SendPassportsToDevice();
  void SendWorkParamsToDevice();
  void SendCalculatedParamsToHost();
  

  int number_of_pipes;

  // На Device:
  // Пасспортные параметры.  
  device_vector<double>   length_dev_vec;
  device_vector<double2>  d_in_out_dev_vec;
  device_vector<double4>  hydr_rough_env_exch_dev_vec;
  // Рабочие параметры.
  device_vector<double2>  p_in_and_t_in_dev_vec;
  device_vector<double>   p_target_dev_vec;  
  device_vector<double>   den_sc_dev_vec;
  device_vector<double>   co2_dev_vec;
  device_vector<double>   n2_dev_vec;
  // Рассчитанные параметры.
  device_vector<double>   q_result_dev_vec;
  device_vector<double>   t_result_dev_vec;
  device_vector<double>   dq_dp_in_result_dev_vec;
  device_vector<double>   dq_dp_out_result_dev_vec;

  // Указатели на память device.
  double *den_sc_dev_;
  double *co2_dev_;
  double *n2_dev_;
  double2*p_in_and_t_in_dev_;
  double *p_target_dev_;
  double *length_dev_;
  double2*d_in_out_dev_;
  double4*hydr_rough_env_exch_dev_;
  double *q_result_dev_;
  double *t_result_dev_;
  double *dq_dp_in_result_dev_;
  double *dq_dp_out_result_dev_;

  // На Host:
  // Пасспортные параметры трубы
  host_vector<double>   length_host;
  host_vector<double2>  d_in_out_host;
  host_vector<double4>  hydr_rough_env_exch_host;
  // Рабочие параметры
  host_vector<double>   den_sc_host;
  host_vector<double>   co2_host;
  host_vector<double>   n2_host;
  host_vector<double2>  p_in_and_t_in_host;
  host_vector<double>   p_target_host;
  // Рассчитанные параметры
  host_vector<double>   q_result_host;
  host_vector<double>   t_result_host;
  host_vector<double>   dq_dp_in_result_host;
  host_vector<double>   dq_dp_out_result_host;
};