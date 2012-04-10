/** \file graph_boost_initial_approx.cpp
Реализация graph_boost_initial_approx.h*/
#include "graph_boost_initial_approx.h"
#include "graph_boost.h"
#include "graph_boost_vertex.h"
#include "graph_boost_edge.h"
#include "graph_boost_engine.h"
// Для работы с opaque итераторми, возвращаемыми графом.
#include <opqit/opaque_iterator.hpp>
// Для std::min
#include <algorithm>

// Для получения объектов труб из рёбер графа.
#include "edge_model_pipe_sequential.h"
#include "model_pipe_sequential.h"

/* Найти максимальное и минимальное паспортное ограничение на давления
среди всех объектов графа.*/
void FindOverallMinAndMaxPressureConstraints(
    GraphBoost *graph, 
    double *overall_p_max,
    double *overall_p_min) {
  /** \todo Реально здесь нужно просто обойти все рёбра. Так же бывает
  нужно обойти все рёбра в топологическом порядке. Нужно сделать итераторы,
  которые будут давать возможность обходоить рёбра графа в топ-м порядке
  и использоавть их здесь, ниже - при задании ограничений, и вообще.*/
  
  // Задаём неправдоподобный p_min, чтобы любой реальный p_min был меньше.
  *overall_p_min = 1000.0;
  *overall_p_max = -1.0;
  for (auto v = graph->VertexBeginNative(); v != graph->VertexEndNative(); 
      ++v) {
    for(auto e = v->OutEdgesBegin(); e != v->OutEdgesEnd(); ++e) {
      *overall_p_max = std::max( *overall_p_max, e->p_max_passport() );
      *overall_p_min = std::min( *overall_p_min, e->p_min_passport() );
    }
  }
}

void SetPressureConstraintsForVertices(
    GraphBoost *graph,
    double overall_p_min,
    double overall_p_max ) {
  /*1. В топологическом порядке "сверху вниз" для каждго узла его ограничение 
  на максимальное давление 
  p_max = min( min( passport.p_max исх-х труб ), min( p_max вх-х узлов ) ).*/
  for(auto v = graph->VertexBeginTopological();
      v != graph->VertexEndTopological(); ++v) { 
    // Задаём неправдоподобный min, чтобы любой реальный был его меньше.
    double min_of_out_p_max( 1000.0 );
    // Вычисляем min(passport.p_max) для всех исходящих из узла рёбер.
    for(auto out_e = v->OutEdgesBegin();
        out_e != v->OutEdgesEnd(); ++out_e) {
      min_of_out_p_max = std::min(
          min_of_out_p_max, out_e->p_max_passport() );
    } // Конец цикла по исходящим рёбрам.
    /* Находим min ( p_max вх-х узлов) - рассчитанные на предыдущих шагах 
    ограничения.*/
    double min_of_in_vertices(1000.0);
    for(auto in_v = v->InVerticesBegin(); in_v != v->inVerticesEnd(); ++in_v) {
      min_of_in_vertices = std::min(min_of_in_vertices, in_v->p_max());
    }
    v->set_p_max( std::min( min_of_out_p_max, min_of_in_vertices ) );
  } // Конец цикла по узлам в топологическом порядке

  /*2. В обратном топологическом порядке "снизу ввверх" для каждого узла его 
  ограничение на мин давление 
  p_min = max( max( passport.p_min вх-х рёбер), ( max( p_min ) исх-х узлов ).*/
  /* Нас устраивает, что первая в топологическом порядке вершина не будет 
  обработана, так как у неё не может быть входов.*/
  bool first_done = false;
  for(auto v = graph->VertexEndTopological() - 1;
      first_done == false; --v) {
    // Вычисляем max(passport.p_min) для входящих в узел рёбер.
    double max_of_in_p_min(-1.0);    
    for(auto in_e = v->InEdgesBegin(); in_e != v->InEdgesEnd(); ++in_e) {
      max_of_in_p_min = std::max(max_of_in_p_min, in_e->p_min_passport());
    } // Конец цикла по входящим в узел рёбрам.
    // Вычисляем max( p_min исх-х узлов).
    double max_of_out_v = -1.0;
    for(auto out_v = v->OutVerticesBegin(); out_v != v->OutVerticesEnd(); 
        ++out_v) {
      max_of_out_v = std::max(max_of_out_v, out_v->p_min());
    }
    v->set_p_min( std::max(max_of_in_p_min, max_of_out_v) );
    if(v == graph->VertexBeginTopological()) {
      first_done = true; // Обработали первую вершину - заканчиваем.
    }
  } // Конец цикла по вершинам в обратном топологическом порядке.
}

bool ChechPressureConstraintsForVertices(
    GraphBoost *graph,
    double overall_p_min,
    double overall_p_max ) {
  //1. overall_p_min, overall_p_max должны быть > 0, p_min <= p_max.
  if(overall_p_max <= 0.0) {
    return false;
  }
  if(overall_p_min <= 0.0) {
    return false;
  }
  if(overall_p_min >= overall_p_max) {
    return false;
  }
  for( auto v = graph->VertexBeginTopological(); 
      v != graph->VertexEndTopological(); ++v ) {
    /* 2. Все p_min, p_max должны быть в интервале 
    [overall_p_min, overall_p_max].*/
    if ( v->p_max() > overall_p_max ) {
      return false;
    }
    if( v->p_min() < overall_p_min ) {
      return false;
    }
    // 3. Для каждого узла интервал [p_min, p_max] должен быть не пустым.
    if( v->p_max() - v->p_min() <= 0) {
      return false;
    }
  } // Конец перебора всех вершин.
  return true; // Если добрались до сюда, значит всё корректно.
}

// Задать для всех входов указанное давление.
void SetPressureForInsAndOuts(GraphBoost *g, double p_in, double p_out) {
  // Все входы в топологическом порядке должны быть сначала?
  for(auto v = g->VertexBeginTopological(); v != g->VertexEndTopological(); 
      ++v) {
    // Если у входа не задано давление задаём p_in.
    if( v->IsGraphInput() == true && v->PIsReady() == false) {
      v->set_p(p_in);
    }
    // Если у выхода не задано давление задаём p_out.
    if( v->IsGraphOutput() == true && v->PIsReady() == false ) {
      v->set_p(p_out);
    }
  }
}

/* Задать давление в исходной вершине, зная путь от неё до вершины с PIsReady.
Путь имеет вид < ( vвх(Pвх), v?(P?) ), (. , .), (. , .), (. , vвых(Pвых) ) >
Формула: P? = sqrt( Pвх^2 - ( x/l ) * (Pвх^2 - Pвых^2) ), где
Pвх - давление на входе пути - должно быть известно, или
расчитано, так как мы идём в топологическом порядке;
x - длина ребра на входе пути
l - общая длина пути (включая x);
Pвых - давление в вершине, явл-ся концом пути.*/
double CountPressureApproxByPath(
    GraphBoost *g,
    std::vector<GraphBoostEdge> *path) {
  // Входящее ребро и его свойства.
  GraphBoostEdge in_e = path->front();
  double x = in_e.pipe_length;
  GraphBoostVertex first_v = g->engine()->graph_[in_e.in_vertex_id()];
  double p_in = first_v.p();
  // Давление на выходе.
  GraphBoostEdge out_e = path->back();
  GraphBoostVertex last_v = g->engine()->graph_[out_e.out_vertex_id()];
  double p_out = last_v.p();
  // Рассчитываем общую длину пути.
  double l = 0.0;
  for(auto e = path->begin(); e != path->end(); ++e) {
    l += e->pipe_length;
  }
  double p_res = 
      sqrt( 
          (p_in*p_in) - 
          (x/l) * ( (p_in*p_in) - (p_out*p_out) ) 
      );
  return p_res;
}

void SetInitialApproxPressures(
    GraphBoost *g,
    double overall_p_max,
    double overall_p_min) {
  SetPressureForInsAndOuts(
      g, 
      overall_p_max, // Если давление на входе не известно - задаём макс.
      overall_p_min); // Если давление на выходе не исзвестго - задаём мин.
  for(auto v = g->VertexBeginTopological(); v != g->VertexEndTopological(); 
      ++v) {
    // Для входов и выходов должно уже быть задано.
    if(v->IsGraphInput() == true || v->IsGraphOutput() == true) {
      continue;
    }
    // Находим min p вх узлов.
    double min_p_v_in = 1000.0;
    for(auto in_v = v->InVerticesBegin(); in_v != v->inVerticesEnd(); ++in_v) {
      min_p_v_in = std::min(min_p_v_in, in_v->p());
    }
    // a) Уточняем ограничение для p_max = min(p_max, min p вх узлов).
    v->set_p_max( std::min( v->p_max(), min_p_v_in ) );

    /* б) Ищем путь до ближайшей вершины с заданным давлением и задаём
    давление в соответствии с p входа, p выхода, длиной пути, длиной трубы.
    Ищем просто в глубину.*/
    std::vector<GraphBoostEdge> path; // Путь до вершины с PIsReady.
    // Помещаем в путь ребро от вершины с известным P до текущей v.
    auto v_cur = v->InVerticesBegin(); // Начало ребра при формировании пути.
    GraphBoostVertex::iter_node v_next = v; // Конец ребра при формир-ии пути.
    /* Пока не задано давление или не выход - в нём должно быть задано.
    А давление у меня пока бывает ещё только во входах...*/
    while(v_cur->IsGraphOutput() == false) {
      path.push_back( 
          g->GetEdge(
              v_cur->id_in_graph(), // id входа.
              v_next->id_in_graph() // id выхода.
          )
      ); // Записали первое ребро пути.
      v_cur = v_next;
      v_next = v_next->OutVerticesBegin(); // Первый попавшийся выход.
    } // Путь сформирован.

    double p_count = CountPressureApproxByPath(
        g, // Указатель на текущий граф.
        &(path) // Путь от текущей вершины до вершины с известным p.
    );
    /* Задаваемое значение должно соответствовать ограничениям.
    Если выходит за рамки [pmin, pmax], то задаём по ближнему краю.*/
    double p_res = std::min( p_count, v->p_max() );
    p_res = std::max( p_res, v->p_min() );
    v->set_p(p_res);
  } // Конец обхода всех вершин в топологическом порядке.
}

/* Задать начальные приближения температур в графе.
Начиная от входов с известным T, температура падает до Tос за 50 км.*/
void SetInitialApproxTemperatures(GraphBoost *g, double t_os) {
  for(auto v = g->VertexBeginTopological(); v != g->VertexEndTopological(); 
      ++v) {
    if(v->gas().work_parameters.t > 0) { // Уже задано
      continue;
    }
    auto v_in = v->InVerticesBegin();
    auto e = g->GetEdge( v_in->id_in_graph(), v->id_in_graph() );
    double l = e.pipe_length;
    double t_in = v_in->t();
    double t_res = std::max( t_os, t_in - (l/50)*(t_in-t_os) );
    v->set_t(t_res);
  }
}