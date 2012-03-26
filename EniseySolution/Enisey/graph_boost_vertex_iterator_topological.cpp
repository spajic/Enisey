#include "graph_boost_vertex_iterator_topological.h"

#include "graph_boost_engine.h"

#include "boost/graph/topological_sort.hpp"

GraphBoostVertexIteratorTopological::GraphBoostVertexIteratorTopological(GraphBoostEngine* graph_engine, bool begin) : graph_engine_(graph_engine)
{
  boost::topological_sort(graph_engine->graph_, std::back_inserter(topological_order_));
  std::reverse(topological_order_.begin(), topological_order_.end());
  if(begin == true)
  {
    index_in_top_order_ = 0;
  }
  else
  {
    // ЗА последним элементом
    index_in_top_order_ = topological_order_.size();
  }		
}
GraphBoostVertexIteratorTopological::~GraphBoostVertexIteratorTopological()
{
}

void GraphBoostVertexIteratorTopological::increment()
{
  ++index_in_top_order_;
}
void GraphBoostVertexIteratorTopological::decrement()
{
  --index_in_top_order_;
}
void GraphBoostVertexIteratorTopological::forward(unsigned int n)
{
  index_in_top_order_ += n;
}
void GraphBoostVertexIteratorTopological::advance(unsigned int n)
{
  index_in_top_order_ += n;
}

int GraphBoostVertexIteratorTopological::distance_to(GraphBoostVertexIteratorTopological const& other) const
{
  return other.index_in_top_order_ - this->index_in_top_order_;
}
bool GraphBoostVertexIteratorTopological::equal(GraphBoostVertexIteratorTopological const& other) const
{
  return other.index_in_top_order_ == this->index_in_top_order_;
}
GraphBoostVertex& GraphBoostVertexIteratorTopological::dereference() const 
{ 
  return graph_engine_->graph_[topological_order_[index_in_top_order_]];
}