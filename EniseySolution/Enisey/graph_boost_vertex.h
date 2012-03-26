#pragma once

#include <list>
#include "gas.h"
#include "choice.h"
#include <opqit/opaque_iterator_fwd.hpp>

class GraphBoostEngine;

class GraphBoostVertex
{
public:
  typedef opqit::opaque_iterator<GraphBoostVertex, opqit::forward> iterator;
  iterator ChildVertexIteratorBegin();
  iterator ChildVertexIteratorEnd();

  GraphBoostVertex();
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

  // Сумма q в поддереве доминаторов - для расчёта q при предварительном
  // анализе графа.
  float q_in_dominators_subtree_;

  Gas gas_;

  std::list<int> in_out_id_list;
};