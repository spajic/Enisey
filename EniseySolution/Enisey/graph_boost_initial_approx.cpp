/** \file graph_boost_initial_approx.cpp
Реализация graph_boost_initial_approx.h*/
#include "graph_boost_initial_approx.h"
#include "graph_boost.h"

void SetPressureConstraintsForVertices(GraphBoost *graph) {
  GraphBoost::iterator it = graph->VertexBeginTopological();
  //1. В топологическом порядку "сверху вниз" для каждго узла его ограничение 
  // на максимальное давление p_max = min(passport.p_max) всех его детей.
  for( ; it != graph->VertexEndTopological(); ++it) {
    // Вычисляем min(passport.p_max) для всех детей узла.
    GraphBoost::iterator child_it = it->ChildVertexIteratorBegin();
    for( ; child_it != it->ChildVertexIteratorEnd(); ++child_it) {

    }
  }
}