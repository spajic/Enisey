/** \file parallel_manager_pipe_cuda.cpp
Реализация менеджера параллельного моделирования труб, 
использующего CUDA.*/
#include "parallel_manager_pipe_cuda.h"
#include "model_pipe_sequential.h"
#include <vector>

void ParallelManagerPipeCUDA::
  TakeUnderControl(std::vector<PassportPipe> const &passports) {
    
}
void ParallelManagerPipeCUDA::
  SetWorkParams(std::vector<WorkParams> const &work_params) {    

}
void ParallelManagerPipeCUDA::CalculateAll() {

}
void ParallelManagerPipeCUDA::
  GetCalculatedParams(std::vector<CalculatedParams> *calculated_params){

}