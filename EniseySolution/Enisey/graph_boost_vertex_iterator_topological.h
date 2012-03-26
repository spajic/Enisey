#pragma once

#include <boost/iterator/iterator_facade.hpp>

#include "graph_boost_vertex.h"

#include <vector>

class GraphBoostEngine;

class GraphBoostVertexIteratorTopological : 
  public boost::iterator_facade <
  GraphBoostVertexIteratorTopological,
  GraphBoostVertex,
  boost::random_access_traversal_tag
  >
{
public:
  GraphBoostVertexIteratorTopological(GraphBoostEngine* graph_engine, bool begin);
  ~GraphBoostVertexIteratorTopological();

private:
  friend class boost::iterator_core_access;

  GraphBoostEngine* graph_engine_;

  // Здесь непосредственно воспользовался тем, что знаю тип итератора в GraphEngine - это просто unisgned int,
  // но не стал брать его оттуда - чтобы не включать его заголовок и не усложнять! (KISS!!!)
  // Наверное, можно было бы сделать аналогично Graph и GraphEngine - 
  // сделать ещё класс обёртку над vertex_descriptor, но лучше проще - не увлекаемся.
  std::vector<unsigned int> topological_order_;
  int index_in_top_order_;

  void increment();
  void decrement();
  void forward(unsigned int n);
  void advance(unsigned int n);
  int distance_to(GraphBoostVertexIteratorTopological const& other) const;

  bool equal(GraphBoostVertexIteratorTopological const& other) const;
  GraphBoostVertex& dereference() const; 
};