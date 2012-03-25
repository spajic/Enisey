#include "manager_edge_model_pipe_sequential.h"

#include <algorithm>

#include <ppl.h>

#include "edge_model_pipe_sequential.h"
#include "passport_pipe.h"

ManagerEdgeModelPipeSequential::ManagerEdgeModelPipeSequential()
{
  max_index_ = 0;
  models_.resize(128000);
  edges_.resize(128000); 
}

Edge* ManagerEdgeModelPipeSequential::CreateEdge(const Passport* passport)
{
  ModelPipeSequential pipe(passport);
  models_[max_index_] = pipe;

  EdgeModelPipeSequential edge;
  edge.pointer_to_model_ = &(models_[max_index_]);

  edges_[max_index_] = edge;
  ++max_index_;
  return &(edges_[max_index_ - 1]);
}

void ManagerEdgeModelPipeSequential::CountAll()
{
  Concurrency::parallel_for_each(models_.begin(), models_.end(), [](ModelPipeSequential &model)
    //std::for_each(models_.begin(), models_.end(), [](ModelPipeSequential &model)
  {
    model.Count();
  } );
}

void ManagerEdgeModelPipeSequential::FinishAddingEdges()
{

}