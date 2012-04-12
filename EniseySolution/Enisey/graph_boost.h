#pragma once

#include <opqit/opaque_iterator_fwd.hpp>
#include <vector>

// Класс, непосредственно представляющий граф BGL, forward-declaration
// Здесь forward-declaration м.б. особенно полезен, чтобы оставить
// возможность включить этот заголовочный файл в *.cu файлы.
// Компилятор Cuda не переваривает код BGL.
class GraphBoostEngine;
class GraphBoostEdge;
class GraphBoostVertex;
class ManagerEdge;

class GraphBoost {
 public:
  typedef opqit::opaque_iterator<GraphBoostVertex, opqit::random> iterator;
  GraphBoost();
  ~GraphBoost();
  /// Проверить, существует ли ребро.
  bool EdgeExists(int in_v_id, int out_v_id);
  /// Получить ссылку на ребро графа по id вх-й и исх-й вершины.
  /// Могут быть параллельные рёбра!
  /// GraphBoostEdge& GetEdge(int in_v_id, int out_v_id);
  /** Тип итератора, разыменование которого даёт ссылку на ребро графа.
  Для получения входящихи устаё и исходящих из узла рёбер.*/
  typedef opqit::opaque_iterator<GraphBoostEdge, opqit::bidir> iter_edge;
  iter_edge ParallelEdgesBegin(int in_v_id, int out_v_id);
  iter_edge ParallelEdgesEnd(int in_v_id, int out_v_id);

  int AddVertex(GraphBoostVertex* graph_boost_vertex);
  void AddEdge(GraphBoostEdge* graph_boost_edge);
  GraphBoostEngine* engine();
  /// Получить ссылку на вершину по id.
  GraphBoostVertex& GetVertex(int v_id);

  iterator VertexBeginNative();
  iterator VertexEndNative();
  iterator VertexBeginTopological();
  iterator VertexEndTopological();

  ManagerEdge* manager();
  void set_manager(ManagerEdge* manager_edge);
private:
  GraphBoostEngine *engine_;
  ManagerEdge* manager_edge_;
  std::vector<unsigned int> topological_ordering;
};