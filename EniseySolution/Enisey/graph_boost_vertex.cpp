/** \file graph_boost_vertex.cpp 
Реализация graph_boost_vertex.h*/
#include "graph_boost_vertex.h"
#include "graph_boost_engine.h"
#include "choice.h"
#include "graph_boost_edge.h"
#include "graph_boost_vertex_child_vertex_iterator.h"
#include "graph_boost_engine.h"

/* Эти заголовки нужны для того, чтобы предоставить итераторы
OutEdgesBegin/End, InEdgesBegin/End.*/
#include <opqit/opaque_iterator.hpp>
#include "boost/iterator/transform_iterator.hpp"

/** EdgeDereferenceFunctor функтор, принимает в качестве параметра
дескриптор ребра графа, а возвращает ссылку на ребро, которую получает
из графа, используя переданный дескриптор.
Схема здесь такая: 
 1. boost::out_edges(vertex, graph) возвращает диапазон из двух итераторов,
    каждый из которых указывает не на ребро, а на дескриптор ребра, по которому
    уже можно получить ссылку на ребро из графа. Это не удобно, т.к. дескриптор
    обычно не нужен, а нужно как раз ребро.
 2. Поэтому с помощью boost::transform_iterator полученный итератор 
    преорбразуется таким образом, чтобы он возвращал ссылку на ребро.
 3. А возвращается он как opaque_iterator<GraphBoostEdge, opqit::bidir>,
    чтобы упростить интерфейс хедера, не вносить туда сведений о конкретном
    типе итератора. Ведь нам достаточно знать, что это обычный итератор по
    рёбрам графа.
 Наследуем EdgeDereferenceFunctor от std::unary_function. Это просто добавляет
 парочку typedef-ов, однако без этого ничего работать не хотело. Всегда стоит
 наследоваться от соотв-го unary/binary_function, если нужно представить свой
 функтор стандартной библиотеке, boost, и т.д.*/
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
/* Далее - однотипные функции для итераторов, первую разберём подробно.*/
GraphBoostVertex::iter_edge GraphBoostVertex::OutEdgesBegin() {
  // Получаем iterator-range на исходящие рёбра из графа.
  boost::graph_traits<GraphBoostEngine::graph_type>::out_edge_iterator 
    out_ei_first, out_ei_last;	
  boost::tie(out_ei_first, out_ei_last) = 
    boost::out_edges(id_in_graph_, engine_->graph_);
  /* Создаём и возвращаем boost::transform_iterator, который возрващается
  в качестве opqit::opaque_iterator<GraphBoostEdge, opqit::bidir>*/
  boost::transform_iterator<
      EdgeDereferenceFunctor, // Функтор, применяемый к результату итератора.
      // Итератор, к результатоу которого применяется функтор.
      GraphBoostEngine::graph_type::out_edge_iterator>
       iter(out_ei_first, // Итератор на начало последовательности.
            EdgeDereferenceFunctor( &(engine_->graph_) ) // Экземпляр функтора.
       );  
  return iter;
}
GraphBoostVertex::iter_edge GraphBoostVertex::OutEdgesEnd() {
  boost::graph_traits<GraphBoostEngine::graph_type>::out_edge_iterator 
    out_ei_first, out_ei_last;	
  boost::tie(out_ei_first, out_ei_last) = 
    boost::out_edges(id_in_graph_, engine_->graph_);
  boost::transform_iterator<
    EdgeDereferenceFunctor, 
    GraphBoostEngine::graph_type::out_edge_iterator>
    iter(out_ei_last, EdgeDereferenceFunctor(&(engine_->graph_)));  
  return iter;
}
GraphBoostVertex::iter_edge GraphBoostVertex::InEdgesBegin() {
  boost::graph_traits<GraphBoostEngine::graph_type>::in_edge_iterator 
    in_ei_first, in_ei_last;	
  boost::tie(in_ei_first, in_ei_last) = 
    boost::in_edges(id_in_graph_, engine_->graph_);
  boost::transform_iterator<
    EdgeDereferenceFunctor, 
    GraphBoostEngine::graph_type::in_edge_iterator>
    iter(in_ei_first, EdgeDereferenceFunctor(&(engine_->graph_)));  
  return iter;
}
GraphBoostVertex::iter_edge GraphBoostVertex::InEdgesEnd() {
  boost::graph_traits<GraphBoostEngine::graph_type>::in_edge_iterator 
    in_ei_first, in_ei_last;	
  boost::tie(in_ei_first, in_ei_last) = 
    boost::in_edges(id_in_graph_, engine_->graph_);
  boost::transform_iterator<
    EdgeDereferenceFunctor, 
    GraphBoostEngine::graph_type::in_edge_iterator>
    iter(in_ei_last, EdgeDereferenceFunctor(&(engine_->graph_)));  
  return iter;
}

/* Итераторы для получения входящих в узел узлов.*/
/* Функтор для разыменования узлов, более подробно описан аналогичный
функтор EdgeDereferenceFunctor.*/
struct NodeDereferenceFunctor : public std::unary_function<
    GraphBoostEngine::graph_type::vertex_descriptor, // Тип аргумента.
    GraphBoostVertex&> { //Тип возрващаемого значения.
  /*g - указатель на граф, из которого по дескриптору нужно будет получать
  искомую ссылку на узел.*/
  NodeDereferenceFunctor(GraphBoostEngine::graph_type *g) : g_(g) { }
  /*Необходимость модификатора const здесь не очевидна, но без него не 
  работало. Стоит поподробнее разобраться с const!*/
  GraphBoostVertex& operator()(
      GraphBoostEngine::graph_type::vertex_descriptor desc) const {
    return (*g_)[desc];
  }
  GraphBoostEngine::graph_type *g_;
};
/* Функтор для получения входящего узла ребра по дескриптору ребра.*/
struct StartNodeOfEdgeFunctor : public std::unary_function<
    GraphBoostEngine::graph_type::edge_descriptor, // Тип аргумента.
    GraphBoostVertex&> { //Тип возрващаемого значения.
  /*g - указатель на граф, из которого по дескриптору нужно будет получать
  искомую ссылку на узел.*/
  StartNodeOfEdgeFunctor(GraphBoostEngine::graph_type *g) : g_(g) { }
  /*Необходимость модификатора const здесь не очевидна, но без него не 
  работало. Стоит поподробнее разобраться с const!*/
  GraphBoostVertex& operator()(
      GraphBoostEngine::graph_type::edge_descriptor edge_desc) const {
    GraphBoostEdge e = (*g_)[edge_desc];
    return (*g_)[e.in_vertex_id()];
  }
  GraphBoostEngine::graph_type *g_;
};
/* Получить входящие узлы для узла не так-то просто! В BGL есть только
adjacent_vertices - это исходящие узлы. Чтобы получить входящие нужно
взять входящие рёбра, и их исходные вершины.*/
GraphBoostVertex::iter_node GraphBoostVertex::InVerticesBegin() {
  // Получаем входящие рёбра.
  boost::graph_traits<GraphBoostEngine::graph_type>::in_edge_iterator
    in_ei_first, in_ei_last;	
  boost::tie(in_ei_first, in_ei_last) = 
    boost::in_edges(id_in_graph_, engine_->graph_);
  boost::transform_iterator<
    StartNodeOfEdgeFunctor, 
    GraphBoostEngine::graph_type::in_edge_iterator>
    iter(in_ei_first, StartNodeOfEdgeFunctor(&(engine_->graph_)));  
  return iter;
}
GraphBoostVertex::iter_node GraphBoostVertex::inVerticesEnd() {
  // Получаем входящие рёбра.
  boost::graph_traits<GraphBoostEngine::graph_type>::in_edge_iterator
    in_ei_first, in_ei_last;	
  boost::tie(in_ei_first, in_ei_last) = 
    boost::in_edges(id_in_graph_, engine_->graph_);
  boost::transform_iterator<
    StartNodeOfEdgeFunctor, 
    GraphBoostEngine::graph_type::in_edge_iterator>
    iter(in_ei_last, StartNodeOfEdgeFunctor(&(engine_->graph_)));  
  return iter;
}

/**\todo Дальнейшие итераторы были реализованы раньше, в рамках проекта London.
Там мне не удалось докопаться до решения, реализованного выше, и там
итераторы реализоавны через самодельные классы-обёртки, реализующие
Boost::IteratorFacade. Очевидно, что это бесполезное монструозное усложнение,
от которого нужно будет избвавиться. Сделать как сделано выше.*/

GraphBoostVertex::iter_node GraphBoostVertex::OutVerticesBegin() {
  return GraphBoostVertexChildVertexIterator(engine_, id_in_graph_, true);
}
GraphBoostVertex::iter_node GraphBoostVertex::OutVerticesEnd() {
  return GraphBoostVertexChildVertexIterator(engine_, id_in_graph_, false);
}



GraphBoostEngine* GraphBoostVertex::engine() { 
  return engine_; 
}
int GraphBoostVertex::id_dominator_in_graph() {
  return id_dominator_in_graph_;
}
void GraphBoostVertex::set_id_dominator_in_graph(int id_dominator_in_graph) {
  id_dominator_in_graph_ = id_dominator_in_graph;
}

GraphBoostVertex::GraphBoostVertex() : 
    engine_(NULL),
    id_in_graph_(-1),
    id_vesta_(-1),
    p_is_ready_(false),
    q_is_ready_(false),
    has_in_out_(false),
    is_graph_input_(false),
    is_graph_output_(false),
    in_out_amount_(0.0),
    amount_in_dominator_subtree_(0.0),
    is_all_children_dominator_(not_init),
    id_dominator_in_graph_(-1),
    id_distant_dominator_(-1),
    q_in_dominators_subtree_(0),
    p_max_(-1),
    p_min_(-1) {
  Gas clean_gas;
  gas_ = clean_gas;
}

int GraphBoostVertex::id_vesta() {
  return id_vesta_;
}
void GraphBoostVertex::set_id_vesta(int id_vesta) {
  id_vesta_ = id_vesta;
}
void GraphBoostVertex::set_id_in_graph(int id_in_graph) {
  id_in_graph_ = id_in_graph;
}
int GraphBoostVertex::id_in_graph() {
  return id_in_graph_;
}

Choice GraphBoostVertex::is_all_children_dominator()
{
  return is_all_children_dominator_;
}
void GraphBoostVertex::set_is_all_children_dominator(Choice choice)
{
  is_all_children_dominator_ = choice;
}
void GraphBoostVertex::set_id_distant_dominator(int id_distant_dominator)
{
  id_distant_dominator_ = id_distant_dominator;
}
int GraphBoostVertex::id_distant_dominator()
{
  return id_distant_dominator_;
}
void GraphBoostVertex::set_q_in_domintator_subtree(float q)
{
  q_in_dominators_subtree_ = q;
}
float GraphBoostVertex::q_in_dominators_subtree()
{
  return q_in_dominators_subtree_;
}

// Отметим, что состояние вершины характеризуется следующими параметрами:	
//	bool has_in_out_; - Имеет хотя бы один вход/выход
//	bool p_is_ready_;	Pадано ли давление в вершине - м.б., если есть вход/выход с заданным P
//  С вершиной может быть связан либо один вход/выход с заданным P, либо 1 или больше входов/выходов c заданным Q
//	bool q_is_ready_; Задан ли расход - м.б. если вершина принадлежит неразвлетвлённому пути от входа/выхода с заданным q
//	bool is_graph_input_; Является ли вершина входом графа.
//	bool is_graph_output_;	Является ли вершина выходом графа.
//	float in_out_amount_; Сумма расходов входов/выходов в узле.

// Различные функции добавления к вершине входов/выходов по своему меняют состояние вершины
// и заносят id входа/выхода из Весты в список данной вершины. 

// Добавляем вход с заданным P - такой можно добавить только один!
// Признак того, что уже добавлен - p_is_ready_ = true;
void GraphBoostVertex::AddInputWithSetP(int id_in_out, float p, float t, GasCompositionReduced composition)
{
  if(PIsReady() == true)
  {
    throw "Can't add more than one inout with set P to vertex!!!";
  }
  else
  {
    has_in_out_ = true;
    in_out_id_list.push_back(id_in_out);
    p_is_ready_ = true;
    gas_.work_parameters.p = p;
    gas_.work_parameters.t = t;
    gas_.composition = composition;
  }
}

// Добавляем вход c заданным Q.
// Если уже был задан вход/выход с заданным P, то этого делать нельзя.
void GraphBoostVertex::AddInputWithSetQ(int id_in_out, float q, float t, GasCompositionReduced composition)
{
  if(PIsReady() == true)
  {
    throw "Can't add more than one inout with set P to vertex!!!";
  }
  else
  {
    has_in_out_ = true;
    in_out_id_list.push_back(id_in_out);
    gas_.work_parameters.q += q;
    gas_.work_parameters.t = t;
    gas_.composition = composition;
    in_out_amount_ += q;
  }
}

// Добавляем выход с заданным P. Вход/выход с заданным P может быть только один.
void GraphBoostVertex::AddOutputWithSetP(int id_in_out, float p)
{
  if(PIsReady() == true)
  {
    throw "Can't add more than one inout with set P to vertex!!!";
  }
  else
  {
    has_in_out_ = true;
    in_out_id_list.push_back(id_in_out);
    p_is_ready_ = true;
    gas_.work_parameters.p = p;
  }
}

// Добавляем выход с заданным Q
void GraphBoostVertex::AddOutputWithSetQ(int id_in_out, float q)
{
  if(PIsReady() == true)
  {
    throw "Can't add more than one inout with set P to vertex!!!";
  }
  else
  {
    has_in_out_ = true;
    in_out_id_list.push_back(id_in_out);
    in_out_amount_ += q;
  }
}

bool GraphBoostVertex::PIsReady()
{
  return p_is_ready_;
}

void GraphBoostVertex::set_engine(GraphBoostEngine* engine)
{
  engine_ = engine;
}

unsigned int GraphBoostVertex::DegreeIn()
{
  return boost::in_degree(id_in_graph_, engine_->graph_);
}
unsigned int GraphBoostVertex::DegreeOut()
{
  return boost::out_degree(id_in_graph_, engine_->graph_);
}
bool GraphBoostVertex::IsGraphInput()
{
  return DegreeIn() == 0;
}
bool GraphBoostVertex::IsGraphOutput()
{
  return DegreeOut() == 0;
}
Gas GraphBoostVertex::gas()
{
  return gas_;
}
bool GraphBoostVertex::HasInOut()
{
  return has_in_out_;
}
float GraphBoostVertex::InOutAmount()
{
  return in_out_amount_;
}

float GraphBoostVertex::p_min() { return p_min_; }
void GraphBoostVertex::set_p_min(float p_min) { p_min_ = p_min;}
float GraphBoostVertex::p_max() { return p_max_; }
void GraphBoostVertex::set_p_max(float p_max) { p_max_ = p_max;}