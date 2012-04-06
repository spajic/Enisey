#include "graph_boost.h"
#include "graph_boost_edge.h"
#include "graph_boost_vertex.h"
#include "manager_edge_model_pipe_sequential.h"

#include <boost/graph/adjacency_list.hpp>

#include "graph_boost_engine.h"

#include "graph_boost_vertex_iterator_native.h"

#include <opqit/opaque_iterator.hpp>

#include "graph_boost_vertex_iterator_topological.h"

GraphBoost::GraphBoost()
{
  engine_ = new GraphBoostEngine;
  manager_ = new ManagerEdgeModelPipeSequential();
}
GraphBoost::~GraphBoost()
{
  delete engine_;
  delete manager_;
}

bool GraphBoost::EdgeExists(int in_v_id, int out_v_id) {
  bool edge_exists(false);
  GraphBoostEngine::graph_type::edge_descriptor edge_desc;
  boost::tie(edge_desc, edge_exists) = 
    boost::edge(
    in_v_id,
    out_v_id,
    engine_->graph_
    );
  return edge_exists;
}
GraphBoostEdge& GraphBoost::GetEdge(int in_v_id, int out_v_id) {
  bool edge_exists(false);
  GraphBoostEngine::graph_type::edge_descriptor edge_desc;
  boost::tie(edge_desc, edge_exists) = 
      boost::edge(
          in_v_id,
          out_v_id,
          engine_->graph_
      );
  return engine_->graph_[edge_desc];
}

GraphBoostEngine* GraphBoost::engine()
{
  return engine_;
}
int GraphBoost::AddVertex(GraphBoostVertex* graph_boost_vertex)
{
  int id_of_created_vertex = boost::add_vertex(engine_->graph_);
  // Получаем из графа ссылку на bundled property для созданной вершины
  // и заполняем его.
  // ToDo: Прояснить такой вопрос. Вроде бы мы получаем ссылку, достаточно
  // ли нам просто с ней работать, чтобы изменения были записаны в граф,
  // или нужно её отдельно ещё раз записать в граф после изменения? - Да, достаточно
  GraphBoostVertex& created_vertex = (engine_->graph_)[id_of_created_vertex];
  graph_boost_vertex->set_id_in_graph(id_of_created_vertex);
  graph_boost_vertex->set_engine(engine_);
  created_vertex = *graph_boost_vertex;
  return id_of_created_vertex;
}

void GraphBoost::AddEdge(GraphBoostEdge* graph_boost_edge)
{
  // Добавляем ребро в граф и полсучаем ссылку на bundled property
  bool result_of_adding_edge;
  boost::graph_traits<GraphBoostEngine::graph_type>::edge_descriptor created_edge_descriptor;
  boost::tie(created_edge_descriptor, result_of_adding_edge) = 
     boost::add_edge(graph_boost_edge->in_vertex_id(), graph_boost_edge->out_vertex_id(), engine_->graph_);
  GraphBoostEdge& created_edge = (engine_->graph_)[created_edge_descriptor];
  // Заполняем свойства и записываем в граф
  created_edge = *graph_boost_edge;
}

GraphBoost::iterator GraphBoost::VertexBeginNative()
{
  return GraphBoostVertexIteratorNative(engine_, true);
}
GraphBoost::iterator GraphBoost::VertexEndNative()
{
  return GraphBoostVertexIteratorNative(engine_, false);
}

GraphBoost::iterator GraphBoost::VertexBeginTopological()
{
  return GraphBoostVertexIteratorTopological(engine_, true);
}
GraphBoost::iterator GraphBoost::VertexEndTopological()
{
  return GraphBoostVertexIteratorTopological(engine_, false);
}

ManagerEdgeModelPipeSequential* GraphBoost::manager() {
  return manager_;
}