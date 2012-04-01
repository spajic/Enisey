/** \file graph_boost_initial_approx.cpp
Реализация graph_boost_initial_approx.h*/
#include "graph_boost_initial_approx.h"
#include "graph_boost.h"
#include "graph_boost_vertex.h"
#include "graph_boost_edge.h"
// Для работы с opaque итераторми, возвращаемыми графом.
#include <opqit/opaque_iterator.hpp>
// Для итератора на дочерние вершины.
#include "graph_boost_vertex_child_vertex_iterator.h"
// Для std::min
#include <algorithm>

void SetPressureConstraintsForVertices(GraphBoost *graph) {
  /*1. В топологическом порядке "сверху вниз" для каждго узла его ограничение 
  на максимальное давление p_max = min(passport.p_max) всех исходящих
  из него рёбер.*/
  for(auto v = graph->VertexBeginTopological(); 
    v != graph->VertexEndTopological(); ++v) {
    // Вычисляем min(passport.p_max) для всех исходящих из узла рёбер.
    float min_of_out_p_max(-1.0);
    for(auto out_e = v->OutEdgesBegin();
        out_e != v->OutEdgesEnd(); ++out_e) {
      if(min_of_out_p_max == -1) {
        min_of_out_p_max = out_e->p_max_passport();
        continue;
      }
      min_of_out_p_max = std::min(
          min_of_out_p_max, out_e->p_max_passport() );
    } // Конец цикла по исходящим рёбрам.
    v->set_p_max(min_of_out_p_max);
  } // Конец цикла по узлам в топологическом порядке.
  /*3. В обратном топологическом порядке "снизу ввверх" для каждого узла его 
  ограничение на мин давление p_min = max(passport.p_min) всех входящих в него
  рёбер.*/
  /* Нас устраивает, что первая в топологическом порядке вершина не будет 
  обработана, так как у неё не может быть входов.*/
  for(auto v = graph->VertexEndTopological() -1 ;
      v != graph->VertexBeginTopological(); --v) {
    // Вычисляем max(passport.p_min) для входящих в узел рёбер.
    float max_of_in_p_min(-1.0);
    for(auto in_e = v->InEdgesBegin(); in_e != v->InEdgesEnd(); ++v) {
      max_of_in_p_min = std::max(max_of_in_p_min, in_e->p_min_passport());
    } // Конец цикла по входящим в узел рёбрам.
    --v;
  } // Конец цикла по вершинам в обратном топологическом порядке.
}