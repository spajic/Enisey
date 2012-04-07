#include "manager_edge_model_pipe_sequential.h"

#include <algorithm>

#include <ppl.h>

#include "edge_model_pipe_sequential.h"
#include "passport_pipe.h"

ManagerEdgeModelPipeSequential::ManagerEdgeModelPipeSequential()
{
  models_.reserve(128);
  edges_.reserve(128);
  /// \todo Взять размер вектора исходя из кол-во объектов ГТС.
  max_index_ = 0;
}

Edge* ManagerEdgeModelPipeSequential::CreateEdge(const Passport* passport)
{
  models_.push_back( ModelPipeSequential(passport) );

  EdgeModelPipeSequential edge;
  edge.pointer_to_model_ = &(models_[max_index_]);
  edges_.push_back(edge);
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