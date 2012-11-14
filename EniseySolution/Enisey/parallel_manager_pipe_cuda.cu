/** \file parallel_manager_pipe_cuda.cpp
Реализация менеджера параллельного моделирования труб, 
использующего CUDA.*/
#include "parallel_manager_pipe_cuda.cuh"
#include "model_pipe_sequential.h"
#include <vector>
// Для векторных типов CUDA double2, double4 и т.д.
#include <vector_types.h>

ParallelManagerPipeCUDA::ParallelManagerPipeCUDA() {
  number_of_pipes = 0;
}
void ParallelManagerPipeCUDA::
  TakeUnderControl(std::vector<PassportPipe> const &passports) { 
    number_of_pipes = passports.size();
    AllocateMemoryOnDevice();
}
void ParallelManagerPipeCUDA::
  SetWorkParams(std::vector<WorkParams> const &work_params) {    

}
void ParallelManagerPipeCUDA::CalculateAll() {

}
void ParallelManagerPipeCUDA::
  GetCalculatedParams(std::vector<CalculatedParams> *calculated_params){

}

void ParallelManagerPipeCUDA::
  AllocateMemoryOnDevice() {
  //cudaMalloc( (void**)&(length_dev_), number_of_pipes * sizeof(double) );
}