/** \file impl_pipe_count_cuda.cu
Функция для вызова расчёта трубы на CUDA - для тестирования.*/
#include "model_pipe_sequential_functions_cuda.cuh"

__global__ 
void FindSequentialQCudaKernel(
    double p_target          , 
    double p_work            , double t_work                      ,  
    double den_sc            , double co2                         , double n2, 
    double d_inner           , double d_outer                     , 
    double roughness_coeff   , double hydraulic_efficiency_coeff  , 
    double t_env             , double heat_exchange_coeff         , 
    double length_of_segment , int number_of_segments             ,
    double*t_dev             , double*q_dev
  ) {
  FindSequentialQCuda(
    p_target          , 
    p_work            , t_work                      ,  
    den_sc            , co2                         , n2, 
    d_inner           , d_outer                     , 
    roughness_coeff   , hydraulic_efficiency_coeff  , 
    t_env             , heat_exchange_coeff         , 
    length_of_segment , number_of_segments          ,
    t_dev             , q_dev
  );
}

extern "C"
void CountQOnDevice(
    double p_target         ,
    double p_work           , double t_work                     ,
    double den_sc           , double co2                        , double n2,
    double d_inner          , double d_outer                    , 
    double roughness_coeff  , double hydraulic_efficiency_coeff , 
    double t_env            , double heat_exchange_coeff        , 
    double length_of_segment, int number_of_segments            , 
    double* t_out           , double* q_out) {
  
  double *q_dev;
  double *t_dev;
  cudaMalloc( (void**) &q_dev, sizeof(double) );
  cudaMalloc( (void**) &t_dev, sizeof(double) );
  
  FindSequentialQCudaKernel<<<1, 1>>>(
    p_target          , 
    p_work            , t_work                      ,  
    den_sc            , co2                         , n2, 
    d_inner           , d_outer                     , 
    roughness_coeff   , hydraulic_efficiency_coeff  , 
    t_env             , heat_exchange_coeff         , 
    length_of_segment , number_of_segments          ,
    t_dev             , q_dev
  );
  cudaMemcpy(q_out, q_dev, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(t_out, t_dev, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(q_dev);
  cudaFree(t_dev);
}