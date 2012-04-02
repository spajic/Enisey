/*Есть два типа предположений: ASSERT(expected, actual) и EXPECT.
Невыполнение ASSERT - фатально, прекращает выполнение теста.
Невыполнение EXPECT - не фатально, выполнение продолжается.

Интересные EXPECT'ы:
FLOAT_EQ, DOUBLE_EQ; — неточное сравнение float, double.
NEAR(val1, val2, abs_error) - c погрешностью abs_error.

Тесты можно группировать по файлам и включить (#include) их в test.cpp.
Сами файлы test_concrete.cpp нужно сделать excluded from build.
Это можно сделать из контекстного меню файла в Solution Explorer.

// The fixture for testing class Foo.
//class FooTest : public ::testing::Test {
//protected:
//  FooTest() {} // You can do set-up work for each test here.
//  virtual ~FooTest() { } // You can do clean-up work that doesn't throw exceptions here.
//  // If the constructor and destructor are not enough for setting up
//  // and cleaning up each test, you can define the following methods:
//  virtual void SetUp() { }
//    // Code here will be called immediately after the constructor (right
//    // before each test).
//  virtual void TearDown() { }
//    // Code here will be called immediately after each test (right
//    // before the destructor).

//  // Objects declared here can be used by all tests in the test case for Foo.
//};
//// Tests that the Foo::Bar() method does Abc.
//TEST_F(FooTest, MethodBarDoesAbc) {
//  const string input_filepath = "this/package/testdata/myinputfile.dat";
//  const string output_filepath = "this/package/testdata/myoutputfile.dat";
//  Foo f;
//  EXPECT_EQ(0, f.Bar(input_filepath, output_filepath));
//}
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

#include "test_functions_gas.cpp"

/* Тест расчёта трубы методом последовательного счёта.
Тестировать будем так - зададим входные данные, получим расчётные рез-ты
и сравним их с рез-ми Весты для таких же входных данных.
Если результаты окажутся похожими, сохраним их и в дальнейшем будем
использовать в качестве эталона.*/
TEST(PipeSequential, CountSequentialOut) {
  // Задаём пасспортные свойства трубы.
  PassportPipe passport;
  FillTestPassportPipe(&passport);
  
  // Задаём свойства газа на входе.
  GasCompositionReduced composition;
  GasWorkParameters params_in;
  composition.density_std_cond = 0.6865365; // [кг/м3]
  composition.co2 = 0;
  composition.n2 = 0;
  params_in.p = 5; // [МПа]
  params_in.t = 293.15; // [К]
  params_in.q = 387.843655734; // [м3/сек]

  // Производим расчёт параметров газа на выходе.
  float p_out(-999.0);
  float t_out(-999.0);
  float number_of_segments = 10;
  float length_of_segment = passport.length_ / number_of_segments;
  /* Получены результаты:
  p_out = 2.9721224, t_out = 280.14999 
  (В Весте - p_out = 2.9769, t_out = 279.78)
  Результаты очень похожи на Весту, принимаем их за эталон.
  Видна особенность - у меня температура выходит на Тос, а в Весте - продолжает падать.
  \todo: разобраться точно с этой особенностью.*/
  FindSequentialOut(
      // Рабочие параметры газового потока на входе.
      params_in.p, params_in.t, params_in.q,  
      // Состав газа.
      composition.density_std_cond, composition.co2, composition.n2, 
      // Паспортные свойства трубы.
      passport.d_inner_, passport.d_outer_, passport.roughness_coeff_, 
      passport.hydraulic_efficiency_coeff_, passport.t_env_, 
      // Свойства внешней среды (тоже входят в пасспорт трубы).
      passport.heat_exchange_coeff_, 
      // Длина сегмента и кол-во сегментов.
      length_of_segment, number_of_segments, 
      // Результаты: out-параметры.
      &t_out, &p_out); 

  float eps = 1.0e-4;
  ASSERT_LE(abs(p_out - 2.9721224), eps);
  ASSERT_LE(abs(t_out - 280.14999), eps);
}

TEST(PipeSequential, Count)
{
  // Тестируем расчёт трубы.
  // Методика тестирования поянтна - сравнить с Вестой, если похоже - 
  // сохранить полученный результат и принять за эталон.
  // Задаём пасспортные свойства трубы.
  PassportPipe passport;
  FillTestPassportPipe(&passport);

  // Задаём свойства газа на входе.
  Gas gas_in;
  FillTestGasIn(&gas_in);

  // задаём параметры газа на выходе
  Gas gas_out;
  FillTestGasOut(&gas_out);

  // Создаём объект трубы
  ModelPipeSequential pipe(&passport);
  pipe.set_gas_in(&gas_in);
  pipe.set_gas_out(&gas_out);
  pipe.Count();

  // Результат Весты для данной конфигурации - q = 387.84
  // результат моего расчёта - 385.8383, что похоже на Весту, но не совпадает.
  // Сохраняем это решение в качестве эталонного - при дальнейших изменениях, тест
  // будет сигнализировать, что что-то изменилось, будем смотреть и разбираться.
  float eps = 0.1;
  ASSERT_LE(abs(pipe.q() - 385.83), eps);
}

TEST(DISABLED_ManagerEdgeModelPipeSequential, LoadTest)
{
  // Создадим в менеджере 50 000 труб, и рассчитаем их 100 раз
  // Подготовим трубу для расчёта
  PassportPipe passport;
  FillTestPassportPipe(&passport);

  // Задаём свойства газа на входе.
  GasCompositionReduced composition;
  composition.density_std_cond = 0.6865365; // [кг/м3]
  composition.co2 = 0;
  composition.n2 = 0;
  GasWorkParameters params_in;
  params_in.p = 5; // [МПа]
  params_in.t = 293.15; // [К]
  params_in.q = 387.843655734; // [м3/сек]
  Gas gas_in;
  gas_in.composition = composition;
  gas_in.work_parameters = params_in;

  // задаём параметры газа на выходе
  Gas gas_out = gas_in;
  gas_out.work_parameters.p = 3;
  gas_out.work_parameters.t = 0;
  gas_out.work_parameters.q = 0;

  Edge *edge;
  ManagerEdgeModelPipeSequential manager;

  // Заполняем менеджер объектами
  for(int i = 50; i > 0; --i)
  {
    edge = manager.CreateEdge(&passport);
    edge->set_gas_in(&gas_in);
    edge->set_gas_out(&gas_out);
  }

  // Имитируем расчёт 10 итераций
  //std::cout << "Performing 10 iterations..." << std::endl;
  for(int i = 1; i <= 1; ++i)
  {
    manager.CountAll();
    std::cout << i << " iterations performed" << std::endl;
  }
}

TEST(LoadFromVesta, MatrixConnectionsLoad) {
  VestaFilesData vsd;
  LoadMatrixConnections(
      "C:\\Enisey\\data\\saratov_gorkiy\\MatrixConnections.dat", &vsd);
  LoadPipeLines(
      "C:\\Enisey\\data\\saratov_gorkiy\\PipeLine.dat", &vsd);
  LoadInOutGRS(
    "C:\\Enisey\\data\\saratov_gorkiy\\InOutGRS.dat", &vsd);
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
TEST(GraphBoostTest, GraphBoostTest)
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

  // Тестируем итераторы
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

  // Тестируем построение дерева доминаторов
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

  // Пробуем итератор обхода дочерних вершин
  GraphBoostVertex v = *(graph.VertexBeginTopological());
  for(GraphBoostVertex::iter_node it = v.OutVerticesBegin(); it != v.OutVerticesEnd(); ++it)
  {
    GraphBoostVertex retrievd_vertex = *it;
  }

}

#include "writer_graphviz.h"
TEST(WriterGraphvizTest, WriterGraphvizTest)
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

TEST(GraphBoost, OutEdgeIteratorTest)
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
  //graph.AddEdge(&edge_3);%
  graph.AddEdge(&edge_4);
  graph.AddEdge(&edge_5);
  graph.AddEdge(&edge_6);
  //graph.AddEdge(&edge_7);
  graph.AddEdge(&edge_8);
  graph.AddEdge(&edge_9);

  // Тестирование OutEdges
  for(auto v_it = graph.VertexBeginTopological(); v_it != graph.VertexEndTopological();
      ++v_it) {
    for(auto e_it = v_it->OutEdgesBegin(); e_it != v_it->OutEdgesEnd(); ++e_it) {
      std::cout << "Edge (" << e_it->in_vertex_id() << ", " << e_it->out_vertex_id() <<
        ")" << std::endl;
    }
  }
  // Тестирование InEdges
  for(auto v_it = graph.VertexBeginTopological()+1; v_it != graph.VertexEndTopological();
    ++v_it) {
      for(auto e_it = v_it->InEdgesBegin(); e_it != v_it->InEdgesEnd(); ++e_it) {
        std::cout << "Edge (" << e_it->in_vertex_id() << ", " << e_it->out_vertex_id() <<
          ")" << std::endl;
      }
  }

  // Тестируем vertex::InVerticesBegin(), InVerticesEnd();
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

#include "graph_boost_load_from_vesta.h"
TEST(LoadGraphFromVestaFiles, LoadGraphFromVestaFiles) {
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
#include "graph_boost_initial_approx.h"
// Тестируем корректность задания поля ограничений в вершинах.
TEST(InitialApprox, CorrectnessOfInitialConstraintsForVertices) {
  // 1. Загружаем схему Саратов-Горький из Весты.
  VestaFilesData vfd;
  LoadMatrixConnections(
    "C:\\Enisey\\data\\saratov_gorkiy\\MatrixConnections.dat", &vfd);
  LoadPipeLines(
    "C:\\Enisey\\data\\saratov_gorkiy\\PipeLine.dat", &vfd);
  LoadInOutGRS(
    "C:\\Enisey\\data\\saratov_gorkiy\\InOutGRS.dat", &vfd);
  GraphBoost graph;
  GraphBoostLoadFromVesta(&graph, &vfd);
  // 2. Рассчитываем overall ограничения по всему графу.
  float overall_p_min(999.0);
  float overall_p_max(-999.0);
  FindOverallMinAndMaxPressureConstraints(
      &graph, 
      &overall_p_max,
      &overall_p_min);
  // 2. Рассчитываем ограничения.
  SetPressureConstraintsForVertices(
      &graph,
      overall_p_min,
      overall_p_max );
  // 3. Проверяем корректность.
  bool ok = ChechPressureConstraintsForVertices(
      &graph,
      overall_p_min,
      overall_p_max);
  EXPECT_EQ(ok, true);
  // 4. Выводим граф в GraphViz - посмотреть.
  WriterGraphviz writer;
  writer.WriteGraphToFile(graph, 
      "C:\\Enisey\\out\\test_pressure_constraints.dot");
}

// Тестируем корректность задания поля ограничений в вершинах.
TEST(InitialApprox, InitialApprox) {
  // 1. Загружаем схему Саратов-Горький из Весты.
  VestaFilesData vfd;
  LoadMatrixConnections(
    "C:\\Enisey\\data\\saratov_gorkiy\\MatrixConnections.dat", &vfd);
  LoadPipeLines(
    "C:\\Enisey\\data\\saratov_gorkiy\\PipeLine.dat", &vfd);
  LoadInOutGRS(
    "C:\\Enisey\\data\\saratov_gorkiy\\InOutGRS.dat", &vfd);
  GraphBoost graph;
  GraphBoostLoadFromVesta(&graph, &vfd);
  // 2. Рассчитываем overall ограничения по всему графу.
  float overall_p_min(999.0);
  float overall_p_max(-999.0);
  FindOverallMinAndMaxPressureConstraints(
    &graph, 
    &overall_p_max,
    &overall_p_min);
  // 2. Рассчитываем ограничения.
  SetPressureConstraintsForVertices(
    &graph,
    overall_p_min,
    overall_p_max );
  // 3. Задаём начальное приближение давлений.
  SetInitialApproxPressures(&graph, overall_p_max, overall_p_min);
  // 4. Задаём начальное приближение температур.
  SetInitialApproxTemperatures(&graph, 278.0);
  // 5. Выводим граф в GraphViz - посмотреть.
  WriterGraphviz writer;
  writer.WriteGraphToFile(graph, 
    "C:\\Enisey\\out\\test_initial_approx.dot");
}


 int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


// От London
/*
// Testing and mocking
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <functional>
#include <list>

// для getch()
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

	// Отлаживаемый вариант. Можно ли отлаживать лямбды, интересно?
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

  // Ожидаем нажатия клавиши и возвращаем результат, с которым 
  // выполнились тесты
  std::cout << "Press any key..." << std::endl;
  _getch();
  return result;
}
Конец London
*/ 