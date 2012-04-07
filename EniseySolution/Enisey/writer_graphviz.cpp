#include "writer_graphviz.h"

#include <boost/graph/graphviz.hpp>
#include <boost/lexical_cast.hpp>

#include "graph_boost.h"
#include "graph_boost_edge.h"
#include "graph_boost_vertex.h"
#include "edge.h"

#include "graph_boost_engine.h"

class MyVertexWriter {
 public:
  MyVertexWriter(GraphBoostEngine::graph_type* graph) : graph_(graph) { }
  void operator()(std::ostream& out, const GraphBoostEngine::graph_type::vertex_descriptor& v) {
    GraphBoostVertex vertex = (*graph_)[v];
    // В процессе работы собираем строку вида [shape = circle, color = blue, label = "hello;\n bye"]
    // Параметры добавляются в список params, метки в список labels
    std::list<std::string> params;
    std::list<std::string> labels;

    if(vertex.IsGraphInput()) {
      params.push_back("shape = square");
    } 
    //if(vertex.PIsReady()) {
      labels.push_back("P = " + boost::lexical_cast<std::string>(vertex.gas().work_parameters.p / 0.0980665).substr(0, 5)  );
      labels.push_back("T = " + boost::lexical_cast<std::string>(-273.15 + vertex.gas().work_parameters.t).substr(0, 5)  );
      labels.push_back("ro = " + 
        boost::lexical_cast<std::string>( vertex.gas().composition.density_std_cond ) 
      );
      labels.push_back("disb=" + 
          boost::lexical_cast<std::string>( vertex.CountDisbalance() ) 
      );
    //}
    if(vertex.is_all_children_dominator() == yes) {
      labels.push_back("Q = " + boost::lexical_cast<std::string>(vertex.q_in_dominators_subtree()).substr(0, 5) );
    }
    if(vertex.HasInOut()) {
      params.push_back("color = deepskyblue1");
    }
    if(abs(vertex.InOutAmount()) > 0) {
      labels.push_back("InOut = " + boost::lexical_cast<std::string>(vertex.InOutAmount()).substr(0, 5));
    }
    if(vertex.is_all_children_dominator() == no) {
      params.push_back("shape = triangle");
    }
    labels.push_back("id = " + boost::lexical_cast<std::string>(vertex.id_in_graph()));
    //labels.push_back("idom = " + boost::lexical_cast<std::string>(vertex.id_dominator_in_graph()));
    // Вывод рассчитанного диапазона ограничений [p_min, p_max].
    labels.push_back("[" + boost::lexical_cast<std::string>( vertex.p_min() ) +
        ", " + boost::lexical_cast<std::string>( vertex.p_max() ) + "]");


    //------------- Закончили добавлять параметры, собираем результат -----
    std::string result_params = "";
    std::for_each(params.begin(), params.end(), [&result_params](std::string s) {
      result_params += s;
      result_params += ", ";
    });
    // Если были добавлены строки, удаляем последнюю запятую
    if(params.size() > 0) {
      result_params.erase(result_params.end() - 2, result_params.end());
    }

    std::string result_label = "";
    std::for_each(labels.begin(), labels.end(), [&result_label](std::string s) {
      result_label += s;
      result_label += ";\\n";
    });
    if(labels.size() > 0) {
      result_label.erase(result_label.end() - 2, result_label.end());
    }

    // Собираем результирующую строку
    std::string result = "[";
    if(params.size() > 0) {
      result += result_params;
    }
    if(labels.size() > 0) {
      if(params.size() > 0) {
        result += ", ";
      }
      result += "label = \"" + result_label + "\"";
    }
    result += "]";
    out << result;
  }
private:
  GraphBoostEngine::graph_type* graph_;
};

class MyEdgeWriter {
public:
  MyEdgeWriter(GraphBoostEngine::graph_type* graph) : graph_(graph) { }
  void operator()(std::ostream& out, const GraphBoostEngine::graph_type::edge_descriptor& e) {
    GraphBoostEdge edge = (*graph_)[e];
    // Выводить по-разному магистрали и отводы
    // Магистрали - чёрного цвета, отводы - цветные
    // edge_type = 0 - магистраль, edge_type = 2 - отвод, 3 - дюкер, или лупинг
    if(edge.pipe_type() == 2) {
      out << "[label = \"q=" +
          boost::lexical_cast<std::string>( edge.edge()->q() ) + 
          "\\n dq_dp_in = " +
          boost::lexical_cast<std::string>( edge.edge()->dq_dp_in() ) + 
          "\\n dq_dp_out = " +
          boost::lexical_cast<std::string>( edge.edge()->dq_dp_out() ) + 
          "\", color = brown1, style=\"dashed\"]";
    }
    if(edge.edge_type() == -1) {
      out << "[label = \"q=" +
        boost::lexical_cast<std::string>( edge.edge()->q() ) + 
        "\", color = blue, style=\"dashed\"]";
    }
    if(edge.pipe_type() == 3) {
      out << "[label = \"q=" +
        boost::lexical_cast<std::string>( edge.edge()->q() ) + 
        "\", color = green]";
    }
    else {
      out << "[label = \"q =" + 
          boost::lexical_cast<std::string>( edge.edge()->q() ) + 
          "\\n dq_dp_in = " +
          boost::lexical_cast<std::string>( edge.edge()->dq_dp_in() ) + 
          "\\n dq_dp_out = " +
          boost::lexical_cast<std::string>( edge.edge()->dq_dp_out() ) + 
          "\"]";
    }
  }
private:
  GraphBoostEngine::graph_type* graph_;
};
// Вывод графа с помощью функции write_graphviz
void WriterGraphviz::WriteGraphToFile(GraphBoost& graph, std::string file_name) {
  std::ofstream out_file_stream(file_name.data());
  boost::write_graphviz(out_file_stream, graph.engine()->graph_, MyVertexWriter(&(graph.engine()->graph_)), MyEdgeWriter(&(graph.engine()->graph_)));
}