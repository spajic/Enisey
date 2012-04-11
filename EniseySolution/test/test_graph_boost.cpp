/** \file test_graph_boost.cpp
Тесты для класса GraphBoost из graph_boost.h*/
#include "graph_boost.h"

#include "gtest/gtest.h"
#include "test_utils.h"

#include "graph_boost_vertex.h"
#include "graph_boost_edge.h"
#include <opqit/opaque_iterator.hpp>

#include <vector>
#include <algorithm>

TEST(GraphBoostTest, ParallelEdgesIterator_SingleEdge) {
  GraphBoost graph_with_single_edge;
  GraphBoostVertex v;
  int id_v_0 = graph_with_single_edge.AddVertex(&v);
  int id_v_1 = graph_with_single_edge.AddVertex(&v);
  GraphBoostEdge single_edge_with_id(id_v_0, id_v_1);
  const int kEdgeId = 999;
  single_edge_with_id.set_edge_id_vesta(kEdgeId);
  graph_with_single_edge.AddEdge(&single_edge_with_id);
  for(auto e = graph_with_single_edge.ParallelEdgesBegin(id_v_0, id_v_1);
      e != graph_with_single_edge.ParallelEdgesEnd(id_v_0, id_v_1);
      ++e ) {
    EXPECT_EQ( kEdgeId, e->edge_id_vesta() );
  }
} 
TEST(GraphBoostTest, ParallelEdgesIterator_NoEdgesAtAll) {
  GraphBoost graph_with_no_edges;
  GraphBoostVertex v;
  int id_v_0 = graph_with_no_edges.AddVertex(&v);
  int id_v_1 = graph_with_no_edges.AddVertex(&v);
  for(auto e = graph_with_no_edges.ParallelEdgesBegin(id_v_0, id_v_1);
    e != graph_with_no_edges.ParallelEdgesEnd(id_v_0, id_v_1);
    ++e ) {
      FAIL() << "Should not have any edges.";
  }
}
TEST(GraphBoostTest, ParallelEdges_ParallelEdges) {
  GraphBoost graph_with_parallel_edges;
  GraphBoostVertex v;
  int v_id_0 = graph_with_parallel_edges.AddVertex(&v);
  int v_id_1 = graph_with_parallel_edges.AddVertex(&v);
  
  GraphBoostEdge edge_with_id_1(v_id_0, v_id_1);
  const int kId1 = 111;
  edge_with_id_1.set_edge_id_vesta(kId1);
  graph_with_parallel_edges.AddEdge(&edge_with_id_1);
  
  GraphBoostEdge edge_with_id_2(v_id_0, v_id_1);
  const int kId2 = 222;
  edge_with_id_2.set_edge_id_vesta(kId2);
  graph_with_parallel_edges.AddEdge(&edge_with_id_2);
  
  std::vector<int> edge_ids;
  for(auto e = graph_with_parallel_edges.ParallelEdgesBegin(v_id_0, v_id_1);
      e != graph_with_parallel_edges.ParallelEdgesEnd(v_id_0, v_id_1);
      ++e ) {
    edge_ids.push_back( e->edge_id_vesta() );
  }
  std::sort( edge_ids.begin(), edge_ids.end() );
  EXPECT_EQ( kId1, edge_ids[0] );
  EXPECT_EQ( kId2, edge_ids[1] );
}