#pragma once

#include <boost/graph/adjacency_list.hpp>

#include "graph_boost_edge.h"
#include "graph_boost_vertex.h"

class GraphBoostEngine
{
public:
  typedef boost::adjacency_list< 
    boost::vecS,			// Вершины хранятся в std::vector 
    boost::vecS,			// Связанные рёбра для каждой вершины хранятся в std::vector
    boost::bidirectionalS,	// Граф двунаправленный, т.е. для каждой вершины
    // есть доступ и к исходящим, и к входящим рёбрам
    GraphBoostVertex,					// Связанное свойство (bundled property) для узлов
    GraphBoostEdge				    // связанное свойство для Ребра
  > graph_type;

  graph_type graph_;
};