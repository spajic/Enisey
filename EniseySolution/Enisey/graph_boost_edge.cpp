/** \file graph_boost_edge.cpp
Реализация graph_boost_edge.h*/
#include "graph_boost_edge.h"
#include "edge.h"

GraphBoostEdge::GraphBoostEdge() : in_vertex_id_(-1), out_vertex_id_(-1) { }
GraphBoostEdge::GraphBoostEdge(int in_vertex_id, int out_vertex_id) : 
    in_vertex_id_(in_vertex_id), out_vertex_id_(out_vertex_id) { }

float GraphBoostEdge::p_min_passport() { return p_min_passport_; }
void GraphBoostEdge::set_p_min_passport(float p_min_passport) {
  p_min_passport_ = p_min_passport;
}
float GraphBoostEdge::p_max_passport() { return p_max_passport_; }
void GraphBoostEdge::set_p_max_passport(float p_max_passport) {
  p_max_passport_ = p_max_passport;
}

Edge* GraphBoostEdge::edge()
{
  return edge_;
}
void GraphBoostEdge::set_edge(Edge* edge)
{
  edge_ = edge;
}
int GraphBoostEdge::in_vertex_id()
{
  return in_vertex_id_;
}
int GraphBoostEdge::out_vertex_id()
{
  return out_vertex_id_;
}
int GraphBoostEdge::edge_id_vesta()
{
  return edge_id_vesta_;
}
void GraphBoostEdge::set_edge_id_vesta(int edge_id_vesta)
{
  edge_id_vesta_ = edge_id_vesta;
}
void GraphBoostEdge::set_edge_type(int edge_type)
{
  edge_type_ = edge_type;
}
int GraphBoostEdge::edge_type()
{
  return edge_type_;
}
int GraphBoostEdge::pipe_type()
{
  return pipe_type_;
}
void GraphBoostEdge::set_pipe_type(int pipe_type)
{
  pipe_type_ = pipe_type;
}