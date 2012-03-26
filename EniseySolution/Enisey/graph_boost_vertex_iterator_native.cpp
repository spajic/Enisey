#include "graph_boost_vertex_iterator_native.h"

#include "graph_boost_engine.h"

GraphBoostVertexIteratorNative::GraphBoostVertexIteratorNative(GraphBoostEngine* graph_engine, bool begin) : graph_engine_(graph_engine)
{
  std::pair <
    boost::graph_traits<GraphBoostEngine::graph_type>::vertex_iterator, 
    boost::graph_traits<GraphBoostEngine::graph_type>::vertex_iterator
  > vertex_iters = boost::vertices(graph_engine->graph_); 


  if(begin == true)
  {
    graph_boost_vertex_iterator_ = *vertex_iters.first;
  }
  else
  {
    graph_boost_vertex_iterator_ = *vertex_iters.second;
  }		
}
GraphBoostVertexIteratorNative::~GraphBoostVertexIteratorNative()
{
}

void GraphBoostVertexIteratorNative::increment()
{
  ++graph_boost_vertex_iterator_;
}
void GraphBoostVertexIteratorNative::decrement()
{
  --graph_boost_vertex_iterator_;
}
void GraphBoostVertexIteratorNative::forward(unsigned int n)
{
  graph_boost_vertex_iterator_ += n;
}
void GraphBoostVertexIteratorNative::advance(unsigned int n)
{
  graph_boost_vertex_iterator_ += n;
}

int GraphBoostVertexIteratorNative::distance_to(GraphBoostVertexIteratorNative const& other) const
{
  return other.graph_boost_vertex_iterator_ - this->graph_boost_vertex_iterator_;
}
bool GraphBoostVertexIteratorNative::equal(GraphBoostVertexIteratorNative const& other) const
{
  return other.graph_boost_vertex_iterator_ == this->graph_boost_vertex_iterator_;
}
GraphBoostVertex& GraphBoostVertexIteratorNative::dereference() const 
{ 
  return graph_engine_->graph_[graph_boost_vertex_iterator_];
}