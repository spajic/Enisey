/** \file graph_boost_edge.h
GraphBoostEdge - �����, ���������� � ����� GraphBoost.*/
#pragma once

// Forward declarations
class Edge;
/**����� GraphBoostEdge �������� � ����� GraphBoost ��� bundled property.
� ������ ���� �������� ������������ � ����������� ����������� ��������:
p_min_passport, p_max_passport. ����� �� ��� �������� ������� �������� ����
�����, ����� ��� ���������� ����������� ��� ������� ��������� �����������,
������� ��� �����.*/
class GraphBoostEdge {
 public:
  GraphBoostEdge();
  GraphBoostEdge(int in_vertex_id, int out_vertex_id);
  /// ��������� ������������ ����������� �������� �� �������� �������.
  float p_min_passport();
  /// ������� ������������ ����������� �������� �� �������� �������.
  void set_p_min_passport(float p_min_passport);
  /// ��������� ������������� ����������� �������� �� �������� �������.
  float p_max_passport();
  /// ������� ������������� ����������� �������� �� �������� �������.
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

  /**\todo ����� ����� - ������ ��� ������� ���. �����-�. �������, ����� �����
  ���� �� ������ - �����������.*/
  float pipe_length;
  
 private:
  float p_min_passport_; ///< ����������� �������� �� �������� �������.
  float p_max_passport_; ///< ������������ �������� �� �������� �������.
  int in_vertex_id_; 
  int out_vertex_id_;
  int edge_id_vesta_;
  int edge_type_;
  int pipe_type_;
  Edge* edge_;
};