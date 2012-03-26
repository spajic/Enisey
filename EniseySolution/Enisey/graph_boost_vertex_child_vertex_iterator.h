#pragma once

#include <boost/iterator/iterator_facade.hpp>

#include "graph_boost_vertex.h"

#include <opqit/opaque_iterator_fwd.hpp>

class GraphBoostEngine;

class GraphBoostVertexChildVertexIterator : 
  public boost::iterator_facade <
  GraphBoostVertexChildVertexIterator,
  GraphBoostVertex,
  boost::forward_traversal_tag
  >
{
public:
  GraphBoostVertexChildVertexIterator(GraphBoostEngine* graph_engine, unsigned int id_vertex, bool begin);
  ~GraphBoostVertexChildVertexIterator();

  GraphBoostVertexChildVertexIterator(const GraphBoostVertexChildVertexIterator& rhs);

  GraphBoostVertexChildVertexIterator& operator=(const GraphBoostVertexChildVertexIterator& rhs);



private:
  friend class boost::iterator_core_access;

  GraphBoostEngine* graph_engine_;

  opqit::opaque_iterator<const unsigned int, opqit::forward>* vertex_iterator_;
  opqit::opaque_iterator<const unsigned int, opqit::forward>* vertex_iterator_end_;

  void increment();
  bool equal(GraphBoostVertexChildVertexIterator const& other) const;
  GraphBoostVertex& dereference() const; 
};