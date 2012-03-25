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
private:
  int max_index_;
  std::vector<ModelPipeSequential> models_;
  std::vector<EdgeModelPipeSequential> edges_;
};