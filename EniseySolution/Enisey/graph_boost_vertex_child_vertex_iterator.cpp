#include "graph_boost_vertex_child_vertex_iterator.h"
#include "graph_boost_engine.h"
#include <opqit/opaque_iterator.hpp>

GraphBoostVertexChildVertexIterator::GraphBoostVertexChildVertexIterator(GraphBoostEngine* graph_engine, unsigned int id_vertex, bool begin) :
graph_engine_(graph_engine)
{

  //vertex_iterator_ = new boost::graph_traits<GraphBoostEngine::graph_type>::adjacency_iterator::adjacency_iterator;
  //vertex_iterator_end_ = new opqit::opaque_iterator<unsigned int, opqit::forward>;


  //boost::tie(vertex_iterator_, vertex_iterator_end_) = boost::adjacent_vertices(id_vertex, graph_engine->graph_);

  std::pair <
    boost::graph_traits<GraphBoostEngine::graph_type>::adjacency_iterator, 
    boost::graph_traits<GraphBoostEngine::graph_type>::adjacency_iterator
  > adj_vertex_iters = boost::adjacent_vertices(id_vertex, graph_engine->graph_); 

  opqit::opaque_iterator<const unsigned int, opqit::forward> vertex_iterator(adj_vertex_iters.first);
  opqit::opaque_iterator<const unsigned int, opqit::forward> vertex_iterator_end(adj_vertex_iters.second);

  vertex_iterator_ = new opqit::opaque_iterator<const unsigned int, opqit::forward>;
  vertex_iterator_end_ = new opqit::opaque_iterator<const unsigned int, opqit::forward>;

  *vertex_iterator_ = vertex_iterator;
  *vertex_iterator_end_ = vertex_iterator_end;

  if(begin == false)
  {
    *vertex_iterator_ = *vertex_iterator_end_;
  }
}

GraphBoostVertexChildVertexIterator::GraphBoostVertexChildVertexIterator(const GraphBoostVertexChildVertexIterator& rhs)
{
  graph_engine_ = rhs.graph_engine_;
  vertex_iterator_ = new opqit::opaque_iterator<const unsigned int, opqit::forward>;
  vertex_iterator_end_ = new opqit::opaque_iterator<const unsigned int, opqit::forward>;

  *vertex_iterator_ = *rhs.vertex_iterator_;
  *vertex_iterator_end_ = *rhs.vertex_iterator_end_;
}

GraphBoostVertexChildVertexIterator& GraphBoostVertexChildVertexIterator::operator=(const GraphBoostVertexChildVertexIterator& rhs)
{
  graph_engine_ = rhs.graph_engine_;
  *vertex_iterator_ = *rhs.vertex_iterator_;
  *vertex_iterator_end_ = *rhs.vertex_iterator_end_;
  return *this; // reference to *this
}

GraphBoostVertexChildVertexIterator::~GraphBoostVertexChildVertexIterator()
{
  delete vertex_iterator_end_;
  delete vertex_iterator_;
}

void GraphBoostVertexChildVertexIterator::increment()
{
  ++(*vertex_iterator_);
}

bool GraphBoostVertexChildVertexIterator::equal(GraphBoostVertexChildVertexIterator const& other) const
{
  return *(other.vertex_iterator_) == *(this->vertex_iterator_);
} 

GraphBoostVertex& GraphBoostVertexChildVertexIterator::dereference() const
{
  return graph_engine_->graph_[**vertex_iterator_];
}

