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
2. Bi -= n->CountDisbalance()
3. Если в узел входит узел i, по ребру (i, n) расход q(i,n),
     Ani += dq_dpi
     Ann += dq_dpn
4. Если из узла выходит узел i, по ребру (n, i) расход q(n,i),
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
    // 2. Bn -= v->CountDisbalance()
    B_[n] -= v->CountDisbalance();
    // 3. Если в узел входит узел i, по ребру (i,n) расход q(i,n)...
    for(auto v_in = v->InVerticesBegin(); v_in != v->inVerticesEnd(); ++v_in) {
      for(auto e_in = // Обход параллельных рёбер (v_in, v).
              g_->ParallelEdgesBegin( v_in->id_in_graph(), v->id_in_graph() );
          e_in != g_->ParallelEdgesEnd(v_in->id_in_graph(), v->id_in_graph());
          ++ e_in) {
        A_[std::make_pair(n, n)] += e_in->edge()->dq_dp_out();
        if(v_in->PIsReady() == false) {
          int i = v_in->slae_row();
          A_[std::make_pair(n, i)] += e_in->edge()->dq_dp_in();        
        } // Конец обхода входящих рёбер.
      } // Конец обхода параллельных рёбер.
    }
    /** \todo Спрятать методы edge за GraphBoostEdge.*/
    //  4. Если из узла выходит узел i, по ребру (n,i) расход q(n,i)...
    for(auto v_out = v->OutVerticesBegin(); v_out != v->OutVerticesEnd(); 
        ++v_out) {
      for(auto e_out = // Перебор параллельных рёбер (v, v_out).
              g_->ParallelEdgesBegin( v->id_in_graph(), v_out->id_in_graph() );
          e_out!= g_->ParallelEdgesEnd(v->id_in_graph(), v_out->id_in_graph());
          ++e_out) {
        A_[std::make_pair(n, n)] -= e_out->edge()->dq_dp_in();  
        if(v_out->PIsReady() == false) {
          int i = v_out->slae_row();
          A_[std::make_pair(n, i)] -= e_out->edge()->dq_dp_out();
        }
      } // Конец перебора параллельных рёбер (v, v_out).
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
void GasTransferSystem::CountNewIteration(double g) {
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
double GasTransferSystem::CountDisbalance() {
  double d(0.0);
  for(auto v = g_->VertexBeginTopological(); v != g_->VertexEndTopological();
      ++v) {
      d += abs( v->CountDisbalance() );
  }
  return d;
}
int GasTransferSystem::GetIntDisbalance() {
  int s = 0;
  for(auto v = g_->VertexBeginTopological(); v != g_->VertexEndTopological();
    ++v) {
      if( v->AcceptableDisbalance(0.1) == false ) {
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
  double overall_p_min(999.0);
  double overall_p_max(-999.0);
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
      for(auto e = // Перебор параллельных рёбер (v_in, v_out).
              g_->ParallelEdgesBegin( v->id_in_graph(), v_out->id_in_graph() );
          e != g_->ParallelEdgesEnd( v->id_in_graph(), v_out->id_in_graph() );
          ++e ) {
        e->edge()->set_gas_in( &( v->gas() ) );
        e->edge()->set_gas_out( &( v_out->gas() ) );
      } // Конец перебора параллельных рёбер.
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