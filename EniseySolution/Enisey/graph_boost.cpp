#include "graph_boost.h"
#include "graph_boost_edge.h"
#include "graph_boost_vertex.h"
#include "manager_edge_model_pipe_sequential.h"

#include <boost/graph/adjacency_list.hpp>

#include "graph_boost_engine.h"

// Для работы с итераторами.
#include "graph_boost_vertex_iterator_native.h"
#include "graph_boost_vertex_iterator_topological.h"
#include <opqit/opaque_iterator.hpp>
#include "boost/iterator/transform_iterator.hpp"
#include "boost/iterator/filter_iterator.hpp"

#include <iostream>
#include <sstream>
#include <boost/lexical_cast.hpp>

GraphBoost::GraphBoost() {
  engine_ = new GraphBoostEngine;
}
GraphBoost::~GraphBoost() {
  delete engine_;
}

/* Выполнить вывод состояния графа в Весту.
Выводим не в файл, а в вектор строк.
Формат файла, который нам надо построить:
Точность расчётра время расчёта (можно 0, 00:00)
Количество рёбер
Цикл по объектам
  Номер объекта
    Давление на входе объекта [Ата]
     Температура на входе [C]
      Плотность газа при с.у. на входе [кг/м3]
       Низшая теплотворная способность газа на входе объекта
        Раcход газа на входе объекта [млн м3/сут]
         Давление на выходе [Ата]
          Температура на выходе [C]
           Плотность с.у. на выходе [кг/м3]
            Низшая теплотв. способн-ть на выходе
             Расход газа на выходе [млн м3/сут]
              Скорость потока на входе (объект - труба)
               Скорость потока на выходе (объект - труба)
                Запас газа в трубе (млн м3) (объект - труба)
                 Признак направления потока (0 - не изменялся) (объект - труба)
                  %CO2(только для КЦ в не зависимости потоковая модель или нет)
                   %N2(только для КЦ в не зависимости потоковая модель или нет)
Конец цикла по объектам
Примечания: если объект - кран, то после параметра "Расход на выходе объекта"
идёт параметр "Признак направления потока" и строка заканчивается.
Параметров относящихся к трубе (объект - труба) нет, если объект - не труба.

Количество поставщиков и потребителей (ГРС)
Цикл по поставщикам и потребителям (ГРС) 
  ID поставщика/потрбителя ГРС
   Номер узла
    Давление в узле [Ата]
     Температура [C]
      Плотность при с.у. [кг/м3]
       Низшая теплотоворная способность газа на входе объекта [кДж/м3]
        Расход газа [млн м3/сут]
         %CO2
          %N2
           Содержание конденсата (влаги) в газе [г/м3]
*/

class PusherToStringsVector {
 public:
  PusherToStringsVector(std::vector<std::string> *strings) :strings_(strings){}

  template<typename T>
  PusherToStringsVector& operator<< (const T& arg) {
    arg_s = boost::lexical_cast<std::string>(arg);
    if(arg_s != "\n") { // накапливаем строку
      s += arg_s; 
    } else { // Ввели символ перевода строки - push строку в вектор.
      strings_->push_back(s);
      s.clear();
    }
    return *this;
  }
 private:
  std::vector<std::string> *strings_;
  std::string s;
  std::string arg_s;
};

void GraphBoost::OutputToVesta(std::vector<std::string> *ResultFile) {
  ResultFile->clear();
  PusherToStringsVector p(ResultFile);
  // Точность и время расчёта - пока задаём нули.
  p << "0 00:00" << "\n"; 
  // Количество рёбер графа.
  p << NumberOfEdges() << "\n";
  // Цикл по рёбрам графа.
  for(auto v = VertexBeginTopological(); v != VertexEndTopological(); ++v) {
    for(auto e = v->OutEdgesBegin(); e != v->OutEdgesEnd(); ++ e) {
      p << e->edge_id_vesta() << " "; // Номер объекта.
      // Газ на входе ребра.
      Gas gas_in = e->edge()->gas_in();
      p << gas_in.work_parameters.p / 0.0980665 << " "; // Pвх [Ата].
      p << gas_in.work_parameters.t - 273.15 << " "; // Твх [C].
      p << gas_in.composition.density_std_cond << " "; // [кг/м3].
      p << 0.0 << " ";// Низшая теплотворная способность газа на входе объекта
      p << gas_in.work_parameters.q * 0.0864 << " "; // [млн м3/сут].
      // Газ на выходе ребра.
      Gas gas_out = e->edge()->gas_out();
      p << gas_out.work_parameters.p / 0.0980665 << " "; // Pвых [Ата].
      p << gas_out.work_parameters.t - 273.13 << " "; // Tвых [C].
      p << gas_out.composition.density_std_cond << " ";  // [кг/м3].
      p << 0 << " "; // Низш теплотворн способн-ть на выходе.      
      p << gas_out.work_parameters.q * 0.0864 << " "; // [млн м3/сут].
      p << 0.0 << " "; // Скорость потока на входе (объект - труба).
      p << 0.0 << " "; // Скорость потока на выходе (объект - труба).
      p << 0.0 << " "; // Запас газа в трубе (млн м3) (объект - труба).
      // Признак направления потока (1 - не изменялся) (объект - труба).
      if(e->edge()->IsReverse() == false) {
        p << "1" << " ";  
      } else {
        p << "0" << " ";
      }
      //p << 0.0 << " "; // %CO2 для КЦ.
      //p << 0.0 << " "; // %N2 для КЦ.
      p << "\n"; // Заканчиваем строку.
    }
  }
  // Поставщики и потребители InOutGRS.
  // Подсчитываем количество вершин с InOutGRS.
  int InOutGRS_num(0);
  for(auto v = VertexBeginTopological(); v != VertexEndTopological(); ++v) {
    if( v->HasInOut() == true ) {
      ++InOutGRS_num;
    }
  }
  p << InOutGRS_num << "\n";
  // В цикле по поставщикам и потребителям.
  for(auto v = VertexBeginTopological(); v != VertexEndTopological(); ++v) {
    if( v->HasInOut() == false) {
      continue;
    }
    // Цикл по InOutGRS вершины.
    /// \todo Уточнить у МСК, может ли быть в узла несколько входов/выходов?
    for(auto i = v->in_out_id_list().begin(); i != v->in_out_id_list().end();
        ++i) {
      // Расход в входе - это сумма расхода по исходящим рёбрам, в выходе - 
      // сумма расхода по входящим рёбрам.
      double q(0.0);
      if( v->IsGraphInput() == true ) {
        q = v->OutcomingAmount();
      } else if ( v->IsGraphOutput() == true ){
        q = -v->IncomingAmount(); // Для выходов q < 0.
      }
      p << *i << " ";// ID поставщика/потрбителя ГРС.
      p << v->id_vesta() << " "; // Номер узла
      Gas v_gas = v->gas();
      p << v_gas.work_parameters.p / 0.0980665 << " "; //[Ата].
      p << v_gas.work_parameters.t - 273.15 << " "; // [C].
      p << v_gas.composition.density_std_cond << " "; // [кг/м3]
      // Низшая теплотоворная способность газа на входе объекта [кДж/м3]
      p << "0.0" << " ";
      p << q * 0.0864 << " "; //Расход [млн м3/сут].
      p << v_gas.composition.co2 << " "; // %CO2.
      p << v_gas.composition.n2 << " "; // %N2.
      p << "0.0" << " "; //Содержание конденсата (влаги) в газе [г/м3]
      p << "\n"; // заканчиваем строку.
    } // Конец цикла по in_out_grs вершины.
  } // Конец цикла по всем вершинам.
}

bool GraphBoost::EdgeExists(int in_v_id, int out_v_id) {
  bool edge_exists(false);
  GraphBoostEngine::graph_type::edge_descriptor edge_desc;
  boost::tie(edge_desc, edge_exists) = 
    boost::edge(
    in_v_id,
    out_v_id,
    engine_->graph_
    );
  return edge_exists;
}
int GraphBoost::NumberOfEdges() {
  return boost::num_edges(engine_->graph_);
}
int GraphBoost::NumberOfVertices() {
  return boost::num_vertices(engine_->graph_);
}

struct EdgeDereferenceFunctor : public std::unary_function<
    GraphBoostEngine::graph_type::edge_descriptor, // Тип аргумента.
    GraphBoostEdge&> { //Тип возрващаемого значения.
  /*g - указатель на граф, из которого по дескриптору нужно будет получать
  искомую ссылку на ребро*/
  EdgeDereferenceFunctor(GraphBoostEngine::graph_type *g) : g_(g) { }
  /*Необходимость модификатора const здесь не очевидна, но без него не 
  работало. Стоит поподробнее разобраться с const!*/
  GraphBoostEdge& operator()(
      GraphBoostEngine::graph_type::edge_descriptor desc) const {
    return (*g_)[desc];
  }
  GraphBoostEngine::graph_type *g_;
};
struct ParallelEdgesFilterPredicate : public std::unary_function<
    GraphBoostEdge&, // Тип аргумента.
    bool> { //Тип возрващаемого значения.
  /*g - указатель на граф, из которого по дескриптору нужно будет получать
  искомую ссылку на ребро.
  out_v_id - идентификатор вершины, которой должно заканчиваться ребро.*/
  ParallelEdgesFilterPredicate(
      GraphBoostEngine::graph_type *g,
      int out_v_id) : g_(g), out_v_id_(out_v_id) { }
  /*Необходимость модификатора const здесь не очевидна, но без него не 
  работало. Стоит поподробнее разобраться с const!*/
  bool operator()(
      GraphBoostEdge& e) const {
    return e.out_vertex_id() == out_v_id_;
  }
  GraphBoostEngine::graph_type *g_;
  int out_v_id_;
};


/** Сделаем итератор для параллельных рёбер (v, v_out), отфильтровав
итератор v.OutEdges по признаку, что конечная вершина = v2.*/
GraphBoost::iter_edge GraphBoost::ParallelEdgesBegin(
    int in_v_id, int out_v_id) {
  return boost::make_filter_iterator(
      ParallelEdgesFilterPredicate( &(engine_->graph_), out_v_id ),
      GetVertex(in_v_id).OutEdgesBegin(),
      GetVertex(in_v_id).OutEdgesEnd() );
}
GraphBoost::iter_edge GraphBoost::ParallelEdgesEnd(
    int in_v_id, int out_v_id) {
      return boost::make_filter_iterator<ParallelEdgesFilterPredicate>(
        ParallelEdgesFilterPredicate( &(engine_->graph_), out_v_id ),
        GetVertex(in_v_id).OutEdgesEnd(),
        GetVertex(in_v_id).OutEdgesEnd() );
}
GraphBoostVertex& GraphBoost::GetVertex(int v_id) {
  return engine_->graph_[v_id];
}

GraphBoostEngine* GraphBoost::engine() {
  return engine_;
}
int GraphBoost::AddVertex(GraphBoostVertex* graph_boost_vertex) {
  int id_of_created_vertex = boost::add_vertex(engine_->graph_);
  // Получаем из графа ссылку на bundled property для созданной вершины
  // и заполняем его.
  // ToDo: Прояснить такой вопрос. Вроде бы мы получаем ссылку, достаточно
  // ли нам просто с ней работать, чтобы изменения были записаны в граф,
  // или нужно её отдельно ещё раз записать в граф после изменения? - Да, достаточно
  GraphBoostVertex& created_vertex = (engine_->graph_)[id_of_created_vertex];
  graph_boost_vertex->set_id_in_graph(id_of_created_vertex);
  graph_boost_vertex->set_engine(engine_);
  created_vertex = *graph_boost_vertex;
  return id_of_created_vertex;
}

void GraphBoost::AddEdge(GraphBoostEdge* graph_boost_edge) {
  // Добавляем ребро в граф и полсучаем ссылку на bundled property
  bool result_of_adding_edge;
  boost::graph_traits<GraphBoostEngine::graph_type>::edge_descriptor created_edge_descriptor;
  boost::tie(created_edge_descriptor, result_of_adding_edge) = 
     boost::add_edge(graph_boost_edge->in_vertex_id(), graph_boost_edge->out_vertex_id(), engine_->graph_);
  GraphBoostEdge& created_edge = (engine_->graph_)[created_edge_descriptor];
  // Заполняем свойства и записываем в граф
  created_edge = *graph_boost_edge;
}
GraphBoost::iterator GraphBoost::VertexBeginNative() {
  return GraphBoostVertexIteratorNative(engine_, true);
}
GraphBoost::iterator GraphBoost::VertexEndNative() {
  return GraphBoostVertexIteratorNative(engine_, false);
}
GraphBoost::iterator GraphBoost::VertexBeginTopological() {
  return GraphBoostVertexIteratorTopological(engine_, true);
}
GraphBoost::iterator GraphBoost::VertexEndTopological() {
  return GraphBoostVertexIteratorTopological(engine_, false);
}
ManagerEdge* GraphBoost::manager() {
  return manager_edge_;
}
void GraphBoost::set_manager(ManagerEdge *manager_edge) {
  manager_edge_ = manager_edge;
}