/** \file graph_boost_vertex.h
Класс GraphBoostVertex - тип узла, хранящегося в графе GraphBoost.*/
#pragma once

#include <list>
#include "gas.h"
#include "choice.h"
#include <opqit/opaque_iterator_fwd.hpp>

// Forward-declarations.
class GraphBoostEngine;
class GraphBoostEdge;

class GraphBoostVertex {
 public:
  GraphBoostVertex();
  /** Расчитать состав газа в вершине, путём смешивания всех входящих в вершину
  рёбер. Дело в том, что мы не меняем состав газа вершины непостредственно. 
  Это делается ей самой на основании смешения входящих газовых потоков, либо
  InOutGRS для входов. Выполнять нужно в топологическом порядке. */
  void MixGasFlowsFromAdjacentEdges();
  /** Взять состав газа из предка. Выполнять в топологическом порядке.*/
  void InitialMix();
  /** Расчёт дисбаланса в вершине. Дисбаланс имеет знак, если больше нуля,
  преобладает вход, меньше нуля - преобладает выход.*/
  double CountDisbalance();
  /** Находится ли дисбаланс в вершине в допустимых рамках. */
  bool AcceptableDisbalance(double const max_disb);
  /** Получение номера строки СЛАУ, соотв-й узлу.*/
  int slae_row();
  /** Задание номера строки СЛАУ, соотв-й узлу.*/
  void set_slae_row(int row);
  /* Тип итератора, разыменование которого даёт ссылку на ребро графа.
  Для получения входящихи устаё и исходящих из узла рёбер.*/
  typedef opqit::opaque_iterator<GraphBoostEdge, opqit::bidir> iter_edge;
  // Функции получения итераторов на исходящие из узла рёбра.
  iter_edge OutEdgesBegin();
  iter_edge OutEdgesEnd();
  // Функции получения итераторов на входящие в узел рёбра.
  iter_edge InEdgesBegin();
  iter_edge InEdgesEnd();
  /* Тип итератора, разыменование которого даёт ссылку на узел графа.
  Для получения входящих и исходящих из узла узлов.*/
  typedef opqit::opaque_iterator<GraphBoostVertex, opqit::forward> iter_node;
  // Функции получения итераторов на исходящие из узла узлы.
  iter_node OutVerticesBegin();
  iter_node OutVerticesEnd();
  // Функции получения итераторов на входящие в узел узлы.
  iter_node InVerticesBegin();
  iter_node inVerticesEnd();
  /** Функции задания и получения давления. Не влияют на PIsReady().
  \todo Может вообще убрать P из Gas. Оно не нужно при смешивании, в отличие от
  всего остального там присутствующего, и создаёт путаницу.*/
  void set_p(double p);
  double p();
  // Функции задания и получения температруы.
  void set_t(double t);
  double t();

  void set_id_in_graph(int id_in_graph);
  int id_in_graph();
  void set_id_vesta(int id_vesta);
  int id_vesta();

  void AddInputWithSetP(int id_inout, double p, double t, GasCompositionReduced composition);
  void AddInputWithSetQ(int id_inout, double q, double t, GasCompositionReduced composition);
  void AddOutputWithSetP(int id_inout, double p);
  void AddOutputWithSetQ(int id_inout, double q);

  bool PIsReady();
  bool QIsReady();
  bool HasInOut();

  unsigned int DegreeIn();
  unsigned int DegreeOut();

  bool IsGraphInput();
  bool IsGraphOutput();

  double InOutAmount();

  void set_engine(GraphBoostEngine* engine);
  GraphBoostEngine* engine();
  Gas gas();

  int id_dominator_in_graph();
  void set_id_dominator_in_graph(int id_dominator_in_graph);

  void set_is_all_children_dominator(Choice choice);
  Choice is_all_children_dominator();

  void set_id_distant_dominator(int id_distant_dominator);
  int id_distant_dominator();
  void set_q_in_domintator_subtree(double q);
  double q_in_dominators_subtree();
// Для вычисления ограничений PMin, PMax с учётом предков и детей.
// Используется при задании начальных приближений.
  double p_min();
  void set_p_min(double p_min);
  double p_max();
  void set_p_max(double p_max);
private:
  int slae_row_;
  Gas gas_;

  GraphBoostEngine* engine_;
  int id_in_graph_;
  int id_vesta_;
  
  bool p_is_ready_;
  bool q_is_ready_;
  bool has_in_out_;
  bool is_graph_input_;
  bool is_graph_output_;
  double in_out_amount_;
  double amount_in_dominator_subtree_;
  Choice is_all_children_dominator_;
  int id_dominator_in_graph_;
  int id_distant_dominator_;

// Для вычисления ограничений PMin, PMax с учётом предков и детей.
  double p_min_;
  double p_max_;

  // Сумма q в поддереве доминаторов - для расчёта q при предварительном
  // анализе графа.
  double q_in_dominators_subtree_;



  std::list<int> in_out_id_list;
};