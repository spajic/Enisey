/** \file impl_gas_count_functions_cuda.cpp
CUDA-реализация тестируемых функций свойств газового потока*/
#include "gas_count_functions_cuda.cuh"

__global__ 
void FindBasicGasPropsKernel(
    double den_sc, double co2, double n2,
    double *t_pc, double *p_pc, double *z_sc, double * r_sc){
  *t_pc = FindTPseudoCriticalCuda     (den_sc, co2, n2);
  *p_pc = FindPPseudoCriticalCuda     (den_sc, co2, n2);
  *z_sc = FindZStandartConditionsCuda (den_sc, co2, n2);
  *r_sc = FindRStandartConditionsCuda (den_sc);
}

extern "C"
void FindBasicGasPropsOnDevice(
    double den_sc, double co2, double n2,
    double *t_pc_out, double *p_pc_out, double *z_sc_out, double *r_sc_out) {
  double *t_pc_dev;
  double *p_pc_dev;
  double *z_sc_dev;
  double *r_sc_dev;
  cudaMalloc((void**) &t_pc_dev, sizeof(double) );
  cudaMalloc((void**) &p_pc_dev, sizeof(double) );
  cudaMalloc((void**) &z_sc_dev, sizeof(double) );
  cudaMalloc((void**) &r_sc_dev, sizeof(double) );

  FindBasicGasPropsKernel<<<1, 1>>>(
      den_sc, co2, n2, 
      t_pc_dev, p_pc_dev, z_sc_dev, r_sc_dev);

  cudaMemcpy(t_pc_out, t_pc_dev, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(p_pc_out, p_pc_dev, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(z_sc_out, z_sc_dev, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(r_sc_out, r_sc_dev, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(t_pc_dev);
  cudaFree(p_pc_dev);
  cudaFree(z_sc_dev);
  cudaFree(r_sc_dev);
}
//---------Свойства газа при рабочих параметрах--------------------------------
struct dev_results_struct{
  double p_reduced;
  double t_reduced;
  double c;
  double di;
  double mju;
  double z;
  double ro;
};

__global__
void FindGasPropsAtWorkParamsKernel(
    double p_work,      double t_work, 
    double p_pc,        double t_pc, 
    double den_sc,      double r_sc,
    dev_results_struct *d) {
  d->p_reduced = FindPReducedCuda(p_work       , p_pc           );
  d->t_reduced = FindTReducedCuda(t_work       , t_pc           );
  d->c         = FindCCuda       (d->t_reduced , d->p_reduced   , r_sc       );
  d->di        = FindDiCuda      (d->p_reduced , d->t_reduced                );
  d->mju       = FindMjuCuda     (d->p_reduced , d->t_reduced                );
  d->z         = FindZCuda       (d->p_reduced , d->t_reduced                );
  d->ro        = FindRoCuda      (den_sc       , p_work         , t_work,d->z);
}

extern "C"
void FindGasPropsAtWorkParamsOnDevice(
    double p_work,      double t_work, 
    double p_pc,        double t_pc, 
    double den_sc,      double r_sc,
    double *p_reduced,  double *t_reduced, 
    double *c,          double *di, 
    double *mju,        double *z, 
    double *ro) {

  dev_results_struct *dev_results;  
  dev_results_struct host_results;
  cudaMalloc( (void**) &dev_results, sizeof(dev_results_struct) );
  
  FindGasPropsAtWorkParamsKernel<<<1, 1>>>(
      p_work, t_work,
      p_pc, t_pc,
      den_sc, r_sc,
      dev_results);

  cudaMemcpy(&host_results, dev_results, sizeof(dev_results_struct), 
      cudaMemcpyDeviceToHost);
  cudaFree(dev_results);

  *p_reduced = host_results.p_reduced;
  *t_reduced = host_results.t_reduced;
  *c         = host_results.c;
  *di        = host_results.di;
  *mju       = host_results.mju;
  *z         = host_results.z;
  *ro        = host_results.ro;
}
//--------------------Свойства газа, зависящие от паспорта трубы -------------
__global__ 
void FindReAndLambdaKernel(
    double q    , double den_sc ,
    double mju  , double d_inner,
    double rough, double hydr   ,
    double *re  , double *lambda) {
  *re     = FindReCuda    (q  , den_sc  , mju   , d_inner);
  *lambda = FindLambdaCuda(*re, d_inner , rough , hydr);
}

extern "C" 
void FindReAndLambdaOnDevice(
    double q    , double den_sc ,
    double mju  , double d_inner,
    double rough, double hydr   ,
    double *re  , double *lambda) {
  double *re_dev;
  double *lambda_dev;
  cudaMalloc( (void**) &re_dev    , sizeof( double ) );
  cudaMalloc( (void**) &lambda_dev, sizeof( double ) );
  
  FindReAndLambdaKernel<<<1, 1>>>(
      q     , den_sc,
      mju   , d_inner,
      rough , hydr,
      re_dev, lambda_dev);
  cudaMemcpy(re     , re_dev    , sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(lambda , lambda_dev, sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree( re_dev     );
  cudaFree( lambda_dev );
}
