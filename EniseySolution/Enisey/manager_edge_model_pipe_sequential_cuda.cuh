#pragma once

#include "manager_edge.h"
#include <vector>
#include "cuda_runtime.h"
//#include "edge_model_pipe_sequential_cuda.cuh"

// forward-declarations
struct Passport;
class Edge;
class EdgeModelPipeSequentialCuda;
class Gas;

struct GpuThreadData
{
  cudaStream_t stream;

  float* length_dev_;
  float2* d_in_out_dev_;
  float4* hydr_rough_env_exch_dev_;
  float2* p_in_and_t_in_dev_;
  float* p_target_dev_;
  float* q_result_dev_;
  float* den_sc_dev_;
  float* co2_dev_;
  float* n2_dev_;

  // Пасспортные параметры трубы
  float* length_;
  float2* d_in_out_;
  float4* hydr_rough_env_exch_;

  // Рабочие параметры
  float* den_sc_;
  float* co2_; 
  float* n2_;
  float2* p_in_and_t_in_;
  float* p_target_;
  float* q_result_;
};

// Менеджер рёбер - труб на Cuda.
// Идея в том, чтобы располагать необходимую для расчётов инф-ю
// по подходу StructureOfArrays. Это должно помочь обеспечить
// высокую производительность на Cuda.
class ManagerEdgeModelPipeSequentialCuda: public ManagerEdge
{
public:
  ManagerEdgeModelPipeSequentialCuda();
  ~ManagerEdgeModelPipeSequentialCuda();
  void CountAll();
  Edge* CreateEdge(const Passport* passport);
  void set_gas_in(const Gas* gas, int index);
  void set_gas_out(const Gas* gas, int index);
  float q(int index);
  void FinishAddingEdges();
private:
  static const int max_count_of_edges = 128000;
  static const int kMaxGpuCount_ = 1;
  int gpu_count_;
  bool finish_adding_edges_;
  int max_index_;
  std::vector<EdgeModelPipeSequentialCuda> edges_;
  // Параметры для потоков GPU
  GpuThreadData thread_data_[kMaxGpuCount_];
};