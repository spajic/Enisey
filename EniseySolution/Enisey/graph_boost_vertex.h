#pragma once

#include <list>
#include "gas.h"
#include "choice.h"
#include <opqit/opaque_iterator_fwd.hpp>

class GraphBoostEngine;
class GraphBoostEdge;

class GraphBoostVertex
{
public:
  GraphBoostVertex();

  typedef opqit::opaque_iterator<GraphBoostVertex, opqit::forward> iterator;
  iterator ChildVertexIteratorBegin();
  iterator ChildVertexIteratorEnd();

  typedef opqit::opaque_iterator<GraphBoostEdge, opqit::forward> iter_edge;
  iter_edge OutEdgesBegin();
  iter_edge OutEdgesEnd();
  iter_edge InEdgesBegin();
  iter_edge InEdgesEnd();

  void set_id_in_graph(int id_in_graph);
  int id_in_graph();
  void set_id_vesta(int id_vesta);
  int id_vesta();

  void AddInputWithSetP(int id_inout, float p, float t, GasCompositionReduced composition);
  void AddInputWithSetQ(int id_inout, float q, float t, GasCompositionReduced composition);
  void AddOutputWithSetP(int id_inout, float p);
  void AddOutputWithSetQ(int id_inout, float q);

  bool PIsReady();
  bool QIsReady();
  bool HasInOut();

  unsigned int DegreeIn();
  unsigned int DegreeOut();

  bool IsGraphInput();
  bool IsGraphOutput();

  float InOutAmount();

  void set_engine(GraphBoostEngine* engine);
  GraphBoostEngine* engine();
  Gas gas();

  int id_dominator_in_graph();
  void set_id_dominator_in_graph(int id_dominator_in_graph);

  void set_is_all_children_dominator(Choice choice);
  Choice is_all_children_dominator();

  void set_id_distant_dominator(int id_distant_dominator);
  int id_distant_dominator();
  void set_q_in_domintator_subtree(float q);
  float q_in_dominators_subtree();
// Для вычисления ограничений PMin, PMax с учётом предков и детей.
// Используется при задании начальных приближений.
  float p_min();
  void set_p_min(float p_min);
  float p_max();
  void set_p_max(float p_max);
private:
  GraphBoostEngine* engine_;
  int id_in_graph_;
  int id_vesta_;

  bool p_is_ready_;
  bool q_is_ready_;
  bool has_in_out_;
  bool is_graph_input_;
  bool is_graph_output_;
  float in_out_amount_;
  float amount_in_dominator_subtree_;
  Choice is_all_children_dominator_;
  int id_dominator_in_graph_;
  int id_distant_dominator_;

// Для вычисления ограничений PMin, PMax с учётом предков и детей.
  float p_min_;
  float p_max_;

  // Сумма q в поддереве доминаторов - для расчёта q при предварительном
  // анализе графа.
  float q_in_dominators_subtree_;

  Gas gas_;

  std::list<int> in_out_id_list;
};