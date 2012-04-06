/** \file gas_transfer_system.cpp
Реализация gas_transfer_system.h*/
#include "gas_transfer_system.h"
#include "graph_boost_vertex.h"
// Для задания начального приближения.
#include "graph_boost_initial_approx.h"
// Для итераторов.
#include <opqit/opaque_iterator.hpp>
// Для загрузки графа из файлов Весты.
#include "loader_vesta.h" 
#include "graph_boost.h"
#include "graph_boost_load_from_vesta.h"
// Для работы с рёбрами графа.
#include "graph_boost_edge.h"
#include "edge.h"
// Для расчёта всех рёбер
#include "manager_edge_model_pipe_sequential.h"
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
  // 5. Протягиваем состав газа от входов к выходам.
  MakeInitialMix();
}
void GasTransferSystem::CountAllEdges() {
  /// \todo Нужен итератор рёбер в топологическом порядке.
  for(auto v = g_->VertexBeginTopological(); v != g_->VertexEndTopological();
      ++v) {
    for( auto v_out = v->OutVerticesBegin(); v_out != v->OutVerticesEnd();
        ++v_out) {
      GraphBoostEdge e = g_->GetEdge( v->id_in_graph(), v_out->id_in_graph() );
      e.edge()->set_gas_in( &( v->gas() ) );
      e.edge()->set_gas_out( &( v_out->gas() ) );
    }
  }
  g_->manager()->CountAll();
}
void GasTransferSystem::MakeInitialMix() {
  for(auto v = g_->VertexBeginTopological(); v != g_->VertexEndTopological();
    ++v) {
    v->InitialMix();
  }
}
void GasTransferSystem::MixVertices() {
  for(auto v = g_->VertexBeginTopological(); v != g_->VertexEndTopological();
      ++v) {
    v->MixGasFlowsFromAdjacentEdges();
  }
}