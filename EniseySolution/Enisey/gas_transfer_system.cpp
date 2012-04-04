/** \file gas_transfer_system.cpp
Реализация gas_transfer_system.h*/
#include "gas_transfer_system.h"
// Для задания начального приближения.
#include "graph_boost_initial_approx.h"

// Для загрузки графа из файлов Весты.
#include "loader_vesta.h" 
#include "graph_boost.h"
#include "graph_boost_load_from_vesta.h"
// Для вывода в GraphViz
#include "writer_graphviz.h"

GasTransferSystem::GasTransferSystem() {
  g_ = new GraphBoost();
}
GasTransferSystem::~GasTransferSystem() {
  delete g_;
}
void GasTransferSystem::LoadFromVestaFiles(std::string const path) {
  VestaFilesData vfd;
  LoadMatrixConnections(path + "MatrixConnections.dat", &vfd);
  LoadPipeLines(path + "PipeLine.dat", &vfd);
  LoadInOutGRS(path + "InOutGRS.dat", &vfd);
  GraphBoostLoadFromVesta(g_, &vfd);
}
void const GasTransferSystem::WriteToGraphviz(std::string const filename) {
  WriterGraphviz writer;
  writer.WriteGraphToFile(*g_, filename);
}
void GasTransferSystem::MakeInitialApprox() {
  float overall_p_min(999.0);
  float overall_p_max(-999.0);
  FindOverallMinAndMaxPressureConstraints(
      g_, 
      &overall_p_max,
      &overall_p_min);
  // 1. Рассчитываем overall ограничения по всему графу.
  SetPressureConstraintsForVertices(
    g_,
    overall_p_min,
    overall_p_max );
  // 3. Задаём начальное приближение давлений.
  SetInitialApproxPressures(g_, overall_p_max, overall_p_min);
  // 4. Задаём начальное приближение температур.
  SetInitialApproxTemperatures(g_, 278.0);

}
void GasTransferSystem::CountAllEdges() {

}