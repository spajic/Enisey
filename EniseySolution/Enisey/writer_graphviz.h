#pragma once
#include <string>
// Forward-declaration
class GraphBoost;
class WriterGraphviz {
public:
  void WriteGraphToFile(GraphBoost& graph, std::string file_name);
};