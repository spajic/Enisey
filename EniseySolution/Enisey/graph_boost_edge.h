#pragma once

// Forward declarations
class Edge;

class GraphBoostEdge {
 public:
  GraphBoostEdge();
  GraphBoostEdge(int in_vertex_id, int out_vertex_id);
  void set_edge(Edge* edge);
  int in_vertex_id();
  int out_vertex_id();
  int edge_id_vesta();
  void set_edge_id_vesta(int edge_id_vesta);
  int edge_type();
  void set_edge_type(int edge_type);

  int pipe_type();
  void set_pipe_type(int pipe_type);
  Edge* edge();
 private:
  int in_vertex_id_;
  int out_vertex_id_;
  int edge_id_vesta_;
  int edge_type_;
  int pipe_type_;
  Edge* edge_;
};