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

void FindMinAndMaxPressureConstraints(GraphBoost *graph, 
    float *overall_p_max,
    float *overall_p_min) {
  /** \todo Реально здесь нужно просто обойти все рёбра. Так же бывает
  нужно обойти все рёбра в топологическом порядке. Нужно сделать итераторы,
  которые будут давать возможность обходоить рёбра графа в топ-м порядке
  и использоавть их здесь, ниже - при задании ограничений, и вообще.*/
  
  // Задаём неправдоподобный p_min, чтобы любой реальный p_min был меньше.
  *overall_p_min = 1000.0;
  *overall_p_max = -1.0;
  for (auto v = graph->VertexBeginNative(); v != graph->VertexEndNative(); 
      ++v) {
    for(auto e = v->OutEdgesBegin(); e != v->OutEdgesEnd(); ++v) {
      *overall_p_max = std::max( *overall_p_max, e->p_max_passport() );
      *overall_p_min = std::min( *overall_p_min, e->p_min_passport() );
    }
  }
}

void SetPressureConstraintsForVertices(GraphBoost *graph) {
  /*1. В топологическом порядке "сверху вниз" для каждго узла его ограничение 
  на максимальное давление p_max = min(passport.p_max) всех исходящих
  из него рёбер.*/
  for(auto v = graph->VertexBeginTopological(); 
    v != graph->VertexEndTopological(); ++v) {
    // Задаём неправдоподобный min, чтобы любой реальный был его меньше.
    float min_of_out_p_max(1000.0);
    // Вычисляем min(passport.p_max) для всех исходящих из узла рёбер.
    for(auto out_e = v->OutEdgesBegin();
        out_e != v->OutEdgesEnd(); ++out_e) {
      min_of_out_p_max = std::min(
          min_of_out_p_max, out_e->p_max_passport() );
    } // Конец цикла по исходящим рёбрам.
    /* Находим min ( p_max вх-х узлов) - рассчитанные на предыдущих шагах 
    ограничения.*/
    // Здесь нам потребуется итератор для вх-х узлов.
    v->set_p_max(min_of_out_p_max);
  } // Конец цикла по узлам в топологическом порядке.
  /*2. В обратном топологическом порядке "снизу ввверх" для каждого узла его 
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