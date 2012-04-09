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
// Для решения СЛАУ.
#include "cvm.h"
// Для отладочной печати.
#include <fstream>

GasTransferSystem::GasTransferSystem() {
  g_ = new GraphBoost();
}
GasTransferSystem::~GasTransferSystem() {
  delete g_;
}
void GasTransferSystem::SetSlaeRowNumsForVertices() {
  int n(0);
  for(auto v = g_->VertexBeginTopological(); v != g_->VertexEndTopological();
      ++v) {
    if(v->PIsReady() == true) {
      v->set_slae_row(-1);
    } else {
      v->set_slae_row(n++);
    }
  }
  // Устанавливаем размер СЛАУ - количество узлов с PIsReady = false.
  // На последней итерации цикла сделано n++, поэтому здесь просто n.
  slae_size_ = n;
}
/**Алгоритм формирования СЛАУ: Для всех узлов с PIsReady = false 
1. Сопоставить узлу номер строки в СЛАУ n.
2. Если в узле есть InOutAmount, то Bn -= InOutAmount.
3. Если в узел входит узел i, по ребру (i, n) расход q(i,n),
     Bn -= q(i,n)
     Ani += dq_dpi
     Ann += dq_dpn
4. Если из узла выходит узел i, по ребру (n, i) расход q(n,i),
     Bn += q(n,i)
     Ani -= dq_dpi
     Ann -= dq_dpn.
5. Если в узле N PIsReady = true(), то dq_dpN = 0 для любого ребра.*/
void GasTransferSystem::FormSlae() {
  A_.clear();
  B_.clear();
  B_.resize(slae_size_);
  for(auto v = g_->VertexBeginTopological(); v != g_->VertexEndTopological(); 
      ++v) {
    if(v->PIsReady() == true) {
      continue;
    }
    int n = v->slae_row();
    if( v->HasInOut() == true ) { // 2. Если в узле есть InOutAmount...
      B_[n] -= v->InOutAmount();
    }
    // 3. Если в узел входит узел i, по ребру (i,n) расход q(i,n)...
    for(auto v_in = v->InVerticesBegin(); v_in != v->inVerticesEnd(); ++v_in) {
      int i = v_in->slae_row();
      auto e_in = g_->GetEdge( v_in->id_in_graph(), v->id_in_graph() );
      B_[n] -= e_in.edge()->q();
      if(v_in->PIsReady() == false) {
        A_[std::make_pair(n, i)] += e_in.edge()->dq_dp_in();        
      }
      A_[std::make_pair(n, n)] += e_in.edge()->dq_dp_out();
      // Конец обхода входящих рёбер.
      /** \todo Спрятать методы edge за GraphBoostEdge. Там же сделать 
      проверку, что для PIsReady производная равна нулю.*/
    }
    //  4. Если из узла выходит узел i, по ребру (n,i) расход q(n,i)...
    for(auto v_out = v->OutVerticesBegin(); v_out != v->OutVerticesEnd(); 
        ++v_out) {
      int i = v_out->slae_row();
      auto e_out = g_->GetEdge( v->id_in_graph(), v_out->id_in_graph() );
      B_[n] += e_out.edge()->q();
      if(v_out->PIsReady() == false) {
        A_[std::make_pair(n, i)] -= e_out.edge()->dq_dp_out();
      }
      A_[std::make_pair(n, n)] -= e_out.edge()->dq_dp_in();  
    } // Конец обхода исходящих рёбер.
  } // Конец обхода вершин графа.
}
// Решить сформированную СЛАУ и найти вектор DeltaP_.
void GasTransferSystem::SolveSlae() {
  // Используем пока библиотеку CVM для решения СЛАУ. 
  // В CVM индексация с единицы.
  // Заполняем матрицу A.
  cvm::srmatrix A(slae_size_);
  for(auto a = A_.begin(); a != A_.end(); ++a) {
    int row = a->first.first;
    int col = a->first.second;
    A(row + 1, col + 1) = a->second;
  }
  // Заполняем вектор B.
  cvm::rvector B(slae_size_);
  for(int n = 0; n < slae_size_; ++n) {
    B(n + 1) = B_[n];
  }
  cvm::rvector DeltaP(slae_size_);
  // Для отладки выведем построенные A, B.
  std::ofstream a_fs("C:\\Enisey\\out\\A.txt");
  a_fs << A;
  std::ofstream b_fs("C:\\Enisey\\out\\B.txt");
  b_fs << B;
  DeltaP.solve(A, B);
  std::ofstream p_fs("C:\\Enisey\\out\\DeltaP.txt");
  p_fs << DeltaP;
  // Копируем решение в DeltaP_.
  DeltaP_.clear();
  DeltaP_.resize(slae_size_);
  /// \todo Инициализацию работу со СЛАУ - в отдельную функцию.
  for(int n = 0; n < slae_size_; ++n) {
    DeltaP_[n] = DeltaP[n + 1];
  }
}
void GasTransferSystem::CountNewIteration(float g) {
  FormSlae();
  SolveSlae();
//  WriteToGraphviz("C:\\Enisey\\out\\MixVertices.dot");
  for(auto v = g_->VertexBeginTopological(); v != g_->VertexEndTopological(); 
      ++v) {
    if(v->PIsReady() == true) {
      continue;
    }
    v->set_p( v->p() + DeltaP_[ v->slae_row() ] * g);
  }
  CountAllEdges();
  MixVertices();
}
float GasTransferSystem::CountDisbalance() {
  float d(0.0);
  for(auto v = g_->VertexBeginTopological(); v != g_->VertexEndTopological();
      ++v) {
      d += v->CountDisbalance();
  }
  return d;
}
int GasTransferSystem::GetIntDisbalance() {
  int s = 0;
  for(auto v = g_->VertexBeginTopological(); v != g_->VertexEndTopological();
    ++v) {
      if( v->CountDisbalance() > 0.1 ) {
        ++s;
      }
  }
  return s;
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