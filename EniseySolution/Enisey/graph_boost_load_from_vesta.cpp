/** \file graph_boost_load_from_vesta.cpp
Реализация graph_boost_load_from_vesta.h*/
#include "graph_boost_load_from_vesta.h"
#include "loader_vesta.h"
#include "graph_boost.h"
#include "graph_boost_vertex.h"
#include "graph_boost_edge.h"
#include "manager_edge_model_pipe_sequential.h"
// Функция, заполняющаяя граф по структуре matrix_connections
void GraphBoostLoadFromVesta(GraphBoost* graph, VestaFilesData *vfd) {
  // 1. Создаём все вершины графа (из vfd->vertices_hash).
  for(auto iter = vfd->vertices_hash.begin(); iter != vfd->vertices_hash.end();
      ++iter) {
    // Получаем для вершины её id Весты, надо его запомнить, чтобы 
    // далее строить рёбра.
    int id_vertex = iter->first;
    VertexData vertex_data = iter->second;
    GraphBoostVertex vertex_to_add;
    vertex_to_add.set_id_vesta(id_vertex);
    vertex_to_add.set_engine(graph->engine());
    // Добавляем входы-выходы в вершину.
    for(auto iter = vertex_data.in_outs.begin(); 
        iter != vertex_data.in_outs.end(); iter++) {
      // Если заданы параметры, относящиеся к составу газа, значит это вход.
      if(iter->den_sc > 0) {
        GasCompositionReduced composition;
        composition.density_std_cond = iter->den_sc;
        composition.co2 = iter->co2;
        composition.n2 = iter->n2;
        // Должно быть задано либо давление, либо расход.
        if(iter->pressure > 0) {
          vertex_to_add.AddInputWithSetP(iter->id_in_out, iter->pressure, 
              iter->temp, composition);
        }
        else if(iter->q > 0) {
          vertex_to_add.AddInputWithSetQ(iter->id_in_out, iter->q, iter->temp, 
              composition);
        }
      }
      // Отборы - должно быть задано либо давление, либо расход.
      else {
        if(iter->pressure > 0) {
          vertex_to_add.AddOutputWithSetP(iter->id_in_out, iter->pressure);
        }
        else if(iter->q < 0) {
          vertex_to_add.AddOutputWithSetQ(iter->id_in_out, iter->q);
        }
      }
    }
    // Добавляем вершину в граф и записываем туда заполненный объект
    // vertex_to_add.
    int added_vertex_id = graph->AddVertex(&vertex_to_add);
    // Сохраняем id в графе в структуре VestaFilesData - чтобы далее заполнить
    // граф рёбрами.
    vfd->vertices_hash[id_vertex].id_graph = added_vertex_id;
  }
  // 2. По информации из edges_hash - для каждого ребра есть начало и конец -
  // заполняем граф рёбрами.
  for(auto iter = vfd->edges_hash.begin(); iter != vfd->edges_hash.end(); 
      iter++) {
    int edge_id_vesta = iter->first;
    EdgeData edge_data = iter->second;
    int id_in_vertex_in_graph = 
        vfd->vertices_hash[edge_data.in_vertex_id].id_graph;
    int id_out_vertex_in_graph = 
        vfd->vertices_hash[edge_data.out_vertex_id].id_graph;
    GraphBoostEdge edge(id_in_vertex_in_graph, id_out_vertex_in_graph);
    edge.set_edge_id_vesta(edge_id_vesta);
    edge.set_edge_type(edge_data.edge_type);
    edge.set_pipe_type(edge_data.pipe_type);
    edge.set_edge(graph->manager()->CreateEdge(&(edge_data.passport)));
    /* Проставляем максимальное и минимальное пасспортное давление как свойства
    ребра.*/
    edge.set_p_max_passport(edge_data.passport.p_max_);
    edge.set_p_min_passport(edge_data.passport.p_min_);
    // Добавляем ребро в граф.
    graph->AddEdge(&edge);
  }
}