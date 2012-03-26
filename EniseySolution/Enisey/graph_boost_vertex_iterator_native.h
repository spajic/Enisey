#pragma once

#include <boost/iterator/iterator_facade.hpp>

#include "graph_boost_vertex.h"

class GraphBoostEngine;

class GraphBoostVertexIteratorNative : 
  public boost::iterator_facade <
  GraphBoostVertexIteratorNative,
  GraphBoostVertex,
  boost::random_access_traversal_tag
  >
{
public:
  GraphBoostVertexIteratorNative(GraphBoostEngine* graph_engine, bool begin);
  ~GraphBoostVertexIteratorNative();

private:
  friend class boost::iterator_core_access;

  GraphBoostEngine* graph_engine_;

  // Здесь непосредственно воспользовался тем, что знаю тип итератора в GraphEngine - это просто unisgned int,
  // но не стал брать его оттуда - чтобы не включать его заголовок и не усложнять! (KISS!!!)
  // Наверное, можно было бы сделать аналогично Graph и GraphEngine - 
  // сделать ещё класс обёртку над vertex_descriptor, но лучше проще - не увлекаемся.
  unsigned int graph_boost_vertex_iterator_;

  void increment();
  void decrement();
  void forward(unsigned int n);
  void advance(unsigned int n);
  int distance_to(GraphBoostVertexIteratorNative const& other) const;

  bool equal(GraphBoostVertexIteratorNative const& other) const;
  GraphBoostVertex& dereference() const; 
};