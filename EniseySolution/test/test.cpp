/*���� ��� ���� �������������: ASSERT(expected, actual) � EXPECT.
������������ ASSERT - ��������, ���������� ���������� �����.
������������ EXPECT - �� ��������, ���������� ������������.

���������� EXPECT'�:
double_EQ, DOUBLE_EQ; � �������� ��������� double, double.
NEAR(val1, val2, abs_error) - c ������������ abs_error.

����� ����� ������������ �� ������ � �������� (#include) �� � test.cpp.
���� ����� test_concrete.cpp ����� ������� excluded from build.
��� ����� ������� �� ������������ ���� ����� � Solution Explorer.
*/

#include "gtest/gtest.h"
#include "test_utils.h"

#include "model_pipe_sequential.h"
#include "gas.h"
#include "functions_pipe.h"
#include "passport_pipe.h"
#include "edge.h"
#include "manager_edge_model_pipe_sequential.h"
#include "loader_vesta.h"

// ����� ������� ������� ������� ����.
#include "test_functions_gas.cpp"
// ����� ������ ModelPipeSequential.
#include "test_model_pipe_sequential.cpp"
// ����� ������ GasTransferSystem.
#include "test_gas_transfer_system.cpp"

/* ���� ������� ����� ������� ����������������� �����.
����������� ����� ��� - ������� ������� ������, ������� ��������� ���-��
� ������� �� � ���-�� ����� ��� ����� �� ������� ������.
���� ���������� �������� ��������, �������� �� � � ���������� �����
������������ � �������� �������.*/
TEST(PipeSequential, CountSequentialOut) {
  // ����� ����������� �������� �����.
  PassportPipe passport;
  FillTestPassportPipe(&passport);
  
  // ����� �������� ���� �� �����.
  GasCompositionReduced composition;
  GasWorkParameters params_in;
  composition.density_std_cond = 0.6865365; // [��/�3]
  composition.co2 = 0;
  composition.n2 = 0;
  params_in.p = 5; // [���]
  params_in.t = 293.15; // [�]
  params_in.q = 387.843655734; // [�3/���]

  // ���������� ������ ���������� ���� �� ������.
  double p_out(-999.0);
  double t_out(-999.0);
  double number_of_segments = 10;
  double length_of_segment = passport.length_ / number_of_segments;
  /* �������� ����������:
  p_out = 2.9721224, t_out = 280.14999 
  (� ����� - p_out = 2.9769, t_out = 279.78)
  ���������� ����� ������ �� �����, ��������� �� �� ������.
  ����� ����������� - � ���� ����������� ������� �� ���, � � ����� - ���������� ������.
  \todo: ����������� ����� � ���� ������������.*/
  FindSequentialOut(
      // ������� ��������� �������� ������ �� �����.
      params_in.p, params_in.t, params_in.q,  
      // ������ ����.
      composition.density_std_cond, composition.co2, composition.n2, 
      // ���������� �������� �����.
      passport.d_inner_, passport.d_outer_, passport.roughness_coeff_, 
      passport.hydraulic_efficiency_coeff_, passport.t_env_, 
      // �������� ������� ����� (���� ������ � �������� �����).
      passport.heat_exchange_coeff_, 
      // ����� �������� � ���-�� ���������.
      length_of_segment, number_of_segments, 
      // ����������: out-���������.
      &t_out, &p_out); 

  double eps = 1.0e-4;
  ASSERT_LE(abs(p_out - 2.9721224), eps);
  ASSERT_LE(abs(t_out - 280.14999), eps);
}

#include "graph_boost.h"
#include "graph_boost_vertex.h"
#include "graph_boost_edge.h"
#include "graph_boost_engine.h"
#include <opqit/opaque_iterator.hpp>
#include <functional>
#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dominator_tree.hpp>

TEST(GraphBoostTest, GraphBoostTest) {
  GraphBoost graph;
  GraphBoostVertex dummy_vertex;

  int id_vertex_0 = graph.AddVertex(&dummy_vertex);
  int id_vertex_1 = graph.AddVertex(&dummy_vertex);
  int id_vertex_2 = graph.AddVertex(&dummy_vertex);
  int id_vertex_3 = graph.AddVertex(&dummy_vertex);
  int id_vertex_4 = graph.AddVertex(&dummy_vertex);
  int id_vertex_5 = graph.AddVertex(&dummy_vertex);
  int id_vertex_6 = graph.AddVertex(&dummy_vertex);

  GraphBoostEdge edge_0(id_vertex_2, id_vertex_3);
  GraphBoostEdge edge_1(id_vertex_5, id_vertex_3);
  GraphBoostEdge edge_2(id_vertex_5, id_vertex_1);
  //GraphBoostEdge edge_3(id_vertex_3, id_vertex_1);
  GraphBoostEdge edge_4(id_vertex_3, id_vertex_0);
  GraphBoostEdge edge_5(id_vertex_1, id_vertex_0);
  GraphBoostEdge edge_6(id_vertex_0, id_vertex_4);
  //GraphBoostEdge edge_7(id_vertex_1, id_vertex_4);
  GraphBoostEdge edge_8(id_vertex_6, id_vertex_2);
  GraphBoostEdge edge_9(id_vertex_6, id_vertex_5);

  graph.AddEdge(&edge_0);
  graph.AddEdge(&edge_1);
  graph.AddEdge(&edge_2);
  //graph.AddEdge(&edge_3);
  graph.AddEdge(&edge_4);
  graph.AddEdge(&edge_5);
  graph.AddEdge(&edge_6);
  //graph.AddEdge(&edge_7);
  graph.AddEdge(&edge_8);
  graph.AddEdge(&edge_9);

  // ��������� ���������
  GraphBoost::iterator iter = graph.VertexBeginNative();

  GraphBoostVertex& ref_vertex = *iter;
  //ref_vertex.set_id_in_graph(100);
  dummy_vertex = *iter;

  dummy_vertex = *(++iter);
  dummy_vertex = *(--iter);

  std::for_each(graph.VertexBeginNative(), graph.VertexEndNative(), [](GraphBoostVertex& v)
  {
    std::cout << v.id_in_graph() << ' ';
  } );
  std::cout << std::endl;

  iter = graph.VertexBeginTopological();
  dummy_vertex = *(++iter);
  dummy_vertex = *(--iter);
  iter = graph.VertexEndTopological();

  std::for_each(graph.VertexBeginTopological(), graph.VertexEndTopological(), [](GraphBoostVertex& v)
  {
    std::cout << v.id_in_graph() << ' ';
  } );
  std::cout << std::endl;

  // ��������� ���������� ������ �����������
  // Lengauer-Tarjan dominator tree algorithm
  typedef boost::graph_traits<GraphBoostEngine::graph_type>::vertex_descriptor Vertex;
  typedef boost::property_map<GraphBoostEngine::graph_type, boost::vertex_index_t>::type IndexMap;
  typedef	boost::iterator_property_map<std::vector<Vertex>::iterator, IndexMap> PredMap;

  std::vector<Vertex> domTreePredVector =
    std::vector<Vertex>(boost::num_vertices(graph.engine()->graph_), 
    boost::graph_traits<GraphBoostEngine::graph_type>::null_vertex()
    );

  IndexMap indexMap(boost::get(boost::vertex_index, graph.engine()->graph_));

  PredMap domTreePredMap =
    boost::make_iterator_property_map(domTreePredVector.begin(), indexMap);

  boost::lengauer_tarjan_dominator_tree(graph.engine()->graph_, id_vertex_6, domTreePredMap); 

  std::vector<int> idom(boost::num_vertices(graph.engine()->graph_));

  boost::graph_traits<GraphBoostEngine::graph_type>::vertex_iterator uItr, uEnd;

  for (boost::tie(uItr, uEnd) = boost::vertices(graph.engine()->graph_); uItr != uEnd; ++uItr)
  {
    if (boost::get(domTreePredMap, *uItr) != boost::graph_traits<GraphBoostEngine::graph_type>::null_vertex())
    {
      idom[boost::get(indexMap, *uItr)] =	boost::get(indexMap, boost::get(domTreePredMap, *uItr));
    }
    else
    {
      idom[boost::get(indexMap, *uItr)] = (std::numeric_limits<int>::max)();
    }
  }

  copy(idom.begin(), idom.end(), std::ostream_iterator<int>(std::cout, " "));
  std::cout << std::endl;

  // ������� �������� ������ �������� ������
  GraphBoostVertex v = *(graph.VertexBeginTopological());
  for(GraphBoostVertex::iter_node it = v.OutVerticesBegin(); it != v.OutVerticesEnd(); ++it)
  {
    GraphBoostVertex retrievd_vertex = *it;
  }

}

TEST(GraphBoost, OutEdgeIteratorTest) {
  GraphBoost graph;
  GraphBoostVertex dummy_vertex;

  int id_vertex_0 = graph.AddVertex(&dummy_vertex);
  int id_vertex_1 = graph.AddVertex(&dummy_vertex);
  int id_vertex_2 = graph.AddVertex(&dummy_vertex);
  int id_vertex_3 = graph.AddVertex(&dummy_vertex);
  int id_vertex_4 = graph.AddVertex(&dummy_vertex);
  int id_vertex_5 = graph.AddVertex(&dummy_vertex);
  int id_vertex_6 = graph.AddVertex(&dummy_vertex);

  GraphBoostEdge edge_0(id_vertex_2, id_vertex_3);
  GraphBoostEdge edge_1(id_vertex_5, id_vertex_3);
  GraphBoostEdge edge_2(id_vertex_5, id_vertex_1);
  //GraphBoostEdge edge_3(id_vertex_3, id_vertex_1);
  GraphBoostEdge edge_4(id_vertex_3, id_vertex_0);
  GraphBoostEdge edge_5(id_vertex_1, id_vertex_0);
  GraphBoostEdge edge_6(id_vertex_0, id_vertex_4);
  //GraphBoostEdge edge_7(id_vertex_1, id_vertex_4);
  GraphBoostEdge edge_8(id_vertex_6, id_vertex_2);
  GraphBoostEdge edge_9(id_vertex_6, id_vertex_5);

  graph.AddEdge(&edge_0);
  graph.AddEdge(&edge_1);
  graph.AddEdge(&edge_2);
  //graph.AddEdge(&edge_3);%
  graph.AddEdge(&edge_4);
  graph.AddEdge(&edge_5);
  graph.AddEdge(&edge_6);
  //graph.AddEdge(&edge_7);
  graph.AddEdge(&edge_8);
  graph.AddEdge(&edge_9);

  // ������������ OutEdges
  for(auto v_it = graph.VertexBeginTopological(); v_it != graph.VertexEndTopological();
      ++v_it) {
    for(auto e_it = v_it->OutEdgesBegin(); e_it != v_it->OutEdgesEnd(); ++e_it) {
      std::cout << "Edge (" << e_it->in_vertex_id() << ", " << e_it->out_vertex_id() <<
        ")" << std::endl;
    }
  }
  // ������������ InEdges
  for(auto v_it = graph.VertexBeginTopological()+1; v_it != graph.VertexEndTopological();
    ++v_it) {
      for(auto e_it = v_it->InEdgesBegin(); e_it != v_it->InEdgesEnd(); ++e_it) {
        std::cout << "Edge (" << e_it->in_vertex_id() << ", " << e_it->out_vertex_id() <<
          ")" << std::endl;
      }
  }

  // ��������� vertex::InVerticesBegin(), InVerticesEnd();
  std::cout << "Testing vertex::InVerticesBegin/End\n";
  for(auto v_it = graph.VertexBeginTopological(); v_it != graph.VertexEndTopological();
    ++v_it) {
      for(auto in_v_it = v_it->InVerticesBegin(); 
          in_v_it != v_it->inVerticesEnd(); ++in_v_it) {
            std::cout << in_v_it->id_in_graph() << " ";
      }
      std::cout << std::endl;
  }
}


#include "writer_graphviz.h"
#include "graph_boost_load_from_vesta.h"
TEST(DISABLED_LoadGraphFromVestaFiles, LoadGraphFromVestaFiles) {
  VestaFilesData vfd;
  LoadMatrixConnections(
    "C:\\Enisey\\data\\saratov_gorkiy\\MatrixConnections.dat", &vfd);
  LoadPipeLines(
    "C:\\Enisey\\data\\saratov_gorkiy\\PipeLine.dat", &vfd);
  LoadInOutGRS(
    "C:\\Enisey\\data\\saratov_gorkiy\\InOutGRS.dat", &vfd);
  GraphBoost graph;
  GraphBoostLoadFromVesta(&graph, &vfd);
  WriterGraphviz writer;
  writer.WriteGraphToFile(graph, "C:\\Enisey\\out\\test_load_from_vesta.dot");
}


TEST(DISABLED_WriterGraphvizTest, WriterGraphvizTest)
{
  GraphBoost graph;
  GraphBoostVertex dummy_vertex;

  int id_vertex_0 = graph.AddVertex(&dummy_vertex);
  int id_vertex_1 = graph.AddVertex(&dummy_vertex);
  int id_vertex_2 = graph.AddVertex(&dummy_vertex);
  int id_vertex_3 = graph.AddVertex(&dummy_vertex);
  int id_vertex_4 = graph.AddVertex(&dummy_vertex);
  int id_vertex_5 = graph.AddVertex(&dummy_vertex);
  int id_vertex_6 = graph.AddVertex(&dummy_vertex);

  GraphBoostEdge edge_0(id_vertex_2, id_vertex_3);
  GraphBoostEdge edge_1(id_vertex_5, id_vertex_3);
  GraphBoostEdge edge_2(id_vertex_5, id_vertex_1);
  //GraphBoostEdge edge_3(id_vertex_3, id_vertex_1);
  GraphBoostEdge edge_4(id_vertex_3, id_vertex_0);
  GraphBoostEdge edge_5(id_vertex_1, id_vertex_0);
  GraphBoostEdge edge_6(id_vertex_0, id_vertex_4);
  //GraphBoostEdge edge_7(id_vertex_1, id_vertex_4);
  GraphBoostEdge edge_8(id_vertex_6, id_vertex_2);
  GraphBoostEdge edge_9(id_vertex_6, id_vertex_5);

  graph.AddEdge(&edge_0);
  graph.AddEdge(&edge_1);
  graph.AddEdge(&edge_2);
  //graph.AddEdge(&edge_3);
  graph.AddEdge(&edge_4);
  graph.AddEdge(&edge_5);
  graph.AddEdge(&edge_6);
  //graph.AddEdge(&edge_7);
  graph.AddEdge(&edge_8);
  graph.AddEdge(&edge_9);
  WriterGraphviz writer;
  writer.WriteGraphToFile(graph, "C:\\Enisey\\out\\test_graphviz.dot");
}

#include "graph_boost_initial_approx.h"
// ��������� ������������ ������� ���� ����������� � ��������.
TEST(DISABLED_InitialApprox, CorrectnessOfInitialConstraintsForVertices) {
  // 1. ��������� ����� �������-������� �� �����.
  VestaFilesData vfd;
  LoadMatrixConnections(
    "C:\\Enisey\\data\\saratov_gorkiy\\MatrixConnections.dat", &vfd);
  LoadPipeLines(
    "C:\\Enisey\\data\\saratov_gorkiy\\PipeLine.dat", &vfd);
  LoadInOutGRS(
    "C:\\Enisey\\data\\saratov_gorkiy\\InOutGRS.dat", &vfd);
  GraphBoost graph;
  GraphBoostLoadFromVesta(&graph, &vfd);
  // 2. ������������ overall ����������� �� ����� �����.
  double overall_p_min(999.0);
  double overall_p_max(-999.0);
  FindOverallMinAndMaxPressureConstraints(
      &graph, 
      &overall_p_max,
      &overall_p_min);
  // 2. ������������ �����������.
  SetPressureConstraintsForVertices(
      &graph,
      overall_p_min,
      overall_p_max );
  // 3. ��������� ������������.
  bool ok = ChechPressureConstraintsForVertices(
      &graph,
      overall_p_min,
      overall_p_max);
  EXPECT_EQ(ok, true);
  // 4. ������� ���� � GraphViz - ����������.
  WriterGraphviz writer;
  writer.WriteGraphToFile(graph, 
      "C:\\Enisey\\out\\test_pressure_constraints.dot");
}

// ��������� ��������� �����������.
TEST(DISABLED_InitialApprox, InitialApprox) {
  // 1. ��������� ����� �������-������� �� �����.
  VestaFilesData vfd;
  LoadMatrixConnections(
    "C:\\Enisey\\data\\saratov_gorkiy\\MatrixConnections.dat", &vfd);
  LoadPipeLines(
    "C:\\Enisey\\data\\saratov_gorkiy\\PipeLine.dat", &vfd);
  LoadInOutGRS(
    "C:\\Enisey\\data\\saratov_gorkiy\\InOutGRS.dat", &vfd);
  GraphBoost graph;
  GraphBoostLoadFromVesta(&graph, &vfd);
  // 2. ������������ overall ����������� �� ����� �����.
  double overall_p_min(999.0);
  double overall_p_max(-999.0);
  FindOverallMinAndMaxPressureConstraints(
    &graph, 
    &overall_p_max,
    &overall_p_min);
  // 3. ������������ �����������.
  SetPressureConstraintsForVertices(
    &graph,
    overall_p_min,
    overall_p_max );
  // 4. ����� ��������� ����������� ��������.
  SetInitialApproxPressures(&graph, overall_p_max, overall_p_min);
  // 5. ����� ��������� ����������� ����������.
  SetInitialApproxTemperatures(&graph, 278.0);
  // 6. ������� ���� � GraphViz - ����������.
  WriterGraphviz writer;
  writer.WriteGraphToFile(graph, 
    "C:\\Enisey\\out\\test_initial_approx.dot");
}


TEST(DISABLED_ManagerEdgeModelPipeSequential, LoadTest)
{
  // �������� � ��������� 50 000 ����, � ���������� �� 100 ���
  // ���������� ����� ��� �������
  PassportPipe passport;
  FillTestPassportPipe(&passport);

  // ����� �������� ���� �� �����.
  GasCompositionReduced composition;
  composition.density_std_cond = 0.6865365; // [��/�3]
  composition.co2 = 0;
  composition.n2 = 0;
  GasWorkParameters params_in;
  params_in.p = 5; // [���]
  params_in.t = 293.15; // [�]
  params_in.q = 387.843655734; // [�3/���]
  Gas gas_in;
  gas_in.composition = composition;
  gas_in.work_parameters = params_in;

  // ����� ��������� ���� �� ������
  Gas gas_out = gas_in;
  gas_out.work_parameters.p = 3;
  gas_out.work_parameters.t = 0;
  gas_out.work_parameters.q = 0;

  Edge *edge;
  ManagerEdgeModelPipeSequential manager;

  // ��������� �������� ���������
  for(int i = 50; i > 0; --i)
  {
    edge = manager.CreateEdge(&passport);
    edge->set_gas_in(&gas_in);
    edge->set_gas_out(&gas_out);
  }

  // ��������� ������ 10 ��������
  //std::cout << "Performing 10 iterations..." << std::endl;
  for(int i = 1; i <= 1; ++i)
  {
    manager.CountAll();
    std::cout << i << " iterations performed" << std::endl;
  }
}

TEST(DISABLED_LoadFromVesta, MatrixConnectionsLoad) {
  VestaFilesData vsd;
  LoadMatrixConnections(
    "C:\\Enisey\\data\\saratov_gorkiy\\MatrixConnections.dat", &vsd);
  LoadPipeLines(
    "C:\\Enisey\\data\\saratov_gorkiy\\PipeLine.dat", &vsd);
  LoadInOutGRS(
    "C:\\Enisey\\data\\saratov_gorkiy\\InOutGRS.dat", &vsd);
}

 int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


// �� London
/*
// Testing and mocking
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <functional>
#include <list>

// ��� getch()
#include "conio.h"

// London
#include "gas.h"
#include "gas_count_functions.cuh"

#include "passport_pipe.h"
#include "manager_edge_model_pipe_sequential.h"
#include "model_pipe_sequential.h"
#include "loader_sardan.h"
#include "model_pipe_sequential_functions_proto.cuh"
#include "manager_edge_model_pipe_sequential_cuda.cuh"

#include "utils.h"

TEST(Cuda, Cuda)
{
	PassportPipe passport;
	FillTestPassportPipe(&passport);
	Gas gas_in;
	FillTestGasIn(&gas_in);
	Gas gas_out;
	FillTestGasOut(&gas_out);

	ManagerEdgeModelPipeSequentialCuda manager_cuda;

	std::vector<Edge*> edges;
	edges.resize(4);

	for(int i = 0; i < 4; i++)
	{
		edges[i] = manager_cuda.CreateEdge(&passport);
	}

	std::for_each(edges.begin(), edges.end(), [gas_in, gas_out](Edge* edge)
	{
		edge->set_gas_in(&gas_in);
		edge->set_gas_out(&gas_out);
	});

	// ������������ �������. ����� �� ���������� ������, ���������?
	//edges[0]->set_gas_in(&gas_in);
	//edges[0]->set_gas_out(&gas_out);

	//edges[1]->set_gas_in(&gas_in);
	//edges[1]->set_gas_out(&gas_out);

	manager_cuda.CountAll();
}

//TEST(LoaderSardan, LoaderSardan)
//{
//	loader LoaderSardan;
//	loader.Load();
//}

int main(int argc, char** argv) {
  // The following line must be executed to initialize Google Mock
  // (and Google Test) before running the tests.
  ::testing::InitGoogleMock(&argc, argv);
  
  int result = RUN_ALL_TESTS();

  // ������� ������� ������� � ���������� ���������, � ������� 
  // ����������� �����
  std::cout << "Press any key..." << std::endl;
  _getch();
  return result;
}
����� London
*/ 