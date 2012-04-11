#include "graph_boost.h"
#include "graph_boost_edge.h"
#include "graph_boost_vertex.h"
#include "manager_edge_model_pipe_sequential.h"

#include <boost/graph/adjacency_list.hpp>

#include "graph_boost_engine.h"

// Для работы с итераторами.
#include "graph_boost_vertex_iterator_native.h"
#include "graph_boost_vertex_iterator_topological.h"
#include <opqit/opaque_iterator.hpp>
#include "boost/iterator/transform_iterator.hpp"
#include "boost/iterator/filter_iterator.hpp"

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

struct EdgeDereferenceFunctor : public std::unary_function<
    GraphBoostEngine::graph_type::edge_descriptor, // Тип аргумента.
    GraphBoostEdge&> { //Тип возрващаемого значения.
  /*g - указатель на граф, из которого по дескриптору нужно будет получать
  искомую ссылку на ребро*/
  EdgeDereferenceFunctor(GraphBoostEngine::graph_type *g) : g_(g) { }
  /*Необходимость модификатора const здесь не очевидна, но без него не 
  работало. Стоит поподробнее разобраться с const!*/
  GraphBoostEdge& operator()(
      GraphBoostEngine::graph_type::edge_descriptor desc) const {
    return (*g_)[desc];
  }
  GraphBoostEngine::graph_type *g_;
};
struct ParallelEdgesFilterPredicate : public std::unary_function<
    GraphBoostEdge&, // Тип аргумента.
    bool> { //Тип возрващаемого значения.
  /*g - указатель на граф, из которого по дескриптору нужно будет получать
  искомую ссылку на ребро.
  out_v_id - идентификатор вершины, которой должно заканчиваться ребро.*/
  ParallelEdgesFilterPredicate(
      GraphBoostEngine::graph_type *g,
      int out_v_id) : g_(g), out_v_id_(out_v_id) { }
  /*Необходимость модификатора const здесь не очевидна, но без него не 
  работало. Стоит поподробнее разобраться с const!*/
  bool operator()(
      GraphBoostEdge& e) const {
    return e.out_vertex_id() == out_v_id_;
  }
  GraphBoostEngine::graph_type *g_;
  int out_v_id_;
};


/** Сделаем итератор для параллельных рёбер (v, v_out), отфильтровав
итератор v.OutEdges по признаку, что конечная вершина = v2.*/
GraphBoost::iter_edge GraphBoost::ParallelEdgesBegin(
    int in_v_id, int out_v_id) {
  return boost::make_filter_iterator(
      ParallelEdgesFilterPredicate( &(engine_->graph_), out_v_id ),
      GetVertex(in_v_id).OutEdgesBegin(),
      GetVertex(in_v_id).OutEdgesEnd() );
}
GraphBoost::iter_edge GraphBoost::ParallelEdgesEnd(
    int in_v_id, int out_v_id) {
      return boost::make_filter_iterator<ParallelEdgesFilterPredicate>(
        ParallelEdgesFilterPredicate( &(engine_->graph_), out_v_id ),
        GetVertex(in_v_id).OutEdgesEnd(),
        GetVertex(in_v_id).OutEdgesEnd() );
}
GraphBoostVertex& GraphBoost::GetVertex(int v_id) {
  return engine_->graph_[v_id];
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