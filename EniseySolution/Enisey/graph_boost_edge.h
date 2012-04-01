/** \file graph_boost_edge.h
GraphBoostEdge - класс, хранящийся в рёбрах GraphBoost.*/
#pragma once

// Forward declarations
class Edge;
/**Класс GraphBoostEdge хранится в рёбрах GraphBoost как bundled property.
В классе есть свойства максимальное и минимальное пасспортное давление:
p_min_passport, p_max_passport. Вроде бы эти свойства присущи объектам всех
типов, нужны для вычисления ограничений при расчёте начальных приближений,
поэтому они здесь.*/
class GraphBoostEdge {
 public:
  GraphBoostEdge();
  GraphBoostEdge(int in_vertex_id, int out_vertex_id);
  /// Получение минимального допустимого давления по паспорту объекта.
  float p_min_passport();
  /// Задание минимального допустипого давления по паспорту объекта.
  void set_p_min_passport(float p_min_passport);
  /// Получение максимального допустимого давления по паспорту объекта.
  float p_max_passport();
  /// Задание максимального допустимого давления по паспорту объекта.
  void set_p_max_passport(float p_max_passport);
  void set_edge(Edge* edge);
  Edge* edge();
  int in_vertex_id();
  int out_vertex_id();
  int edge_id_vesta();
  void set_edge_id_vesta(int edge_id_vesta);
  int edge_type();
  void set_edge_type(int edge_type);

  int pipe_type();
  void set_pipe_type(int pipe_type);

  /**\todo Длина трубы - сделал для задания нач. прибл-й. Конечно, здась этого
  быть не должно - разобраться.*/
  float pipe_length;
  
 private:
  float p_min_passport_; ///< Минимальное давление по паспорту объекта.
  float p_max_passport_; ///< Максимальное давление по паспорту объекта.
  int in_vertex_id_; 
  int out_vertex_id_;
  int edge_id_vesta_;
  int edge_type_;
  int pipe_type_;
  Edge* edge_;
};