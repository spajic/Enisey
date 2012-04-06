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
class ManagerEdgeModelPipeSequential;
// class GraphBoostVertexIteratorNative;	// Обёртка над итератором, предоставляемым ф-ей vertices()
// class GraphBoostVertexIteratorTopological; // Перестановка вершин в топологическом порядке

class GraphBoost {
 public:
  typedef opqit::opaque_iterator<GraphBoostVertex, opqit::random> iterator;
  GraphBoost();
  ~GraphBoost();
  /// Проверить, существует ли ребро.
  bool EdgeExists(int in_v_id, int out_v_id);
  /// Получить ссылку на ребро графа по id вх-й и исх-й вершины.
  GraphBoostEdge& GetEdge(int in_v_id, int out_v_id);

  int AddVertex(GraphBoostVertex* graph_boost_vertex);
  void AddEdge(GraphBoostEdge* graph_boost_edge);
  GraphBoostEngine* engine();

  iterator VertexBeginNative();
  iterator VertexEndNative();

  iterator VertexBeginTopological();
  iterator VertexEndTopological();

  //GraphBoostVertexIteratorNative* VertexBeginNative();
  //GraphBoostVertexIteratorNative* VertexEndNative();

  //GraphBoostVertexIteratorTopological* VertexBeginTopological();
  //GraphBoostVertexIteratorTopological* VertexEndTopological();
  ManagerEdgeModelPipeSequential* manager();
private:
  GraphBoostEngine *engine_;
  std::vector<unsigned int> topological_ordering;
  // GraphBoostVertexIteratorNative* vertex_begin_native_;

  ManagerEdgeModelPipeSequential* manager_;
};