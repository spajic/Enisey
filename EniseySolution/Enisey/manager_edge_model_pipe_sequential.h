#pragma once

#include <vector>

#include "edge.h"
#include "manager_edge.h"
#include "model_pipe_sequential.h"
#include "edge_model_pipe_sequential.h"

// forward-declarations
struct Passport;			

class ManagerEdgeModelPipeSequential: public ManagerEdge
{
public:
  ManagerEdgeModelPipeSequential();

  void CountAll();

  Edge* CreateEdge(const Passport* passport);
  void FinishAddingEdges();

  /** Для выгрузки эталонного примера вектора труб и рез-в расчёта*/
  void SavePassportsToFile(std::string file_name);
  void SaveWorkParamsToFile(std::string file_name);
  void SaveCalculatedParamsToFile(std::string file_name);   
private:
  int max_index_;
  std::vector<ModelPipeSequential> models_;
  std::vector<EdgeModelPipeSequential> edges_;
};