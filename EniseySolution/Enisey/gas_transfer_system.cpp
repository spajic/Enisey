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
#include "slae_solver_i.h"
#include "slae_solver_cvm.h"
// Для отладочной печати.
#include <fstream>
#include <vector>
#include <fstream>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>

#include "shiny.h"

GasTransferSystem::GasTransferSystem() {
//LOG4CPLUS_INFO(log_, "Construcor");
  g_ = new GraphBoost();
//LOG4CPLUS_INFO(log_, "End_Construcor");
}
GasTransferSystem::~GasTransferSystem() {
//LOG4CPLUS_INFO(log_, "Destrucor");
  delete g_;
//LOG4CPLUS_INFO(log_, "End_Destrucor");
}

/// Выполнить балансировку системы.
void GasTransferSystem::PeroformBalancing(
    const std::vector<std::string> &MatrixConnectionsFile,
    const std::vector<std::string> &InOutGRSFile,
    const std::vector<std::string> &PipeLinesFile,
    std::vector<std::string> *ResultFile,
    std::vector<double> *AbsDisbalances,
    std::vector<int> *IntDisbalances) {
LOG4CPLUS_INFO(log_, "PerformBalancing");
PROFILE_FUNC();
//LOG4CPLUS_INFO(log_, "After PROFILE_FUNC");
  ManagerEdge* manager_edge_model_pipe_sequential = 
      new ManagerEdgeModelPipeSequential;
  set_manager_edge(manager_edge_model_pipe_sequential);

  SlaeSolverCVM *slae_solver_cvm = new SlaeSolverCVM;
  set_slae_solver(slae_solver_cvm);

  LoadFromVestaFiles(
      MatrixConnectionsFile,
      InOutGRSFile,
      PipeLinesFile);

  MakeInitialApprox();
  CopyGasStateFromVerticesToEdges();
  CountAllEdges();
  MixVertices();

  double d = CountDisbalance();
  AbsDisbalances->push_back(d);
  IntDisbalances->push_back( GetIntDisbalance() );
  double g = 1.0 / 2.0;
  double d_prev = 1000000;
  SetSlaeRowNumsForVertices();
  for(int n = 0; n < 35; ++n) {
    CountNewIteration(g);
    d = CountDisbalance();
    AbsDisbalances->push_back(d);
    if(d_prev < d) {
      g *= 0.9;
    }
    d_prev = d;
    int d_int = GetIntDisbalance();
    IntDisbalances->push_back(d_int);
    if(d_int == 0) {
      break;
    }
  }

  g_->OutputToVesta(ResultFile);

  delete manager_edge_model_pipe_sequential;
  delete slae_solver_cvm;
LOG4CPLUS_INFO(log_, "End_PerformBalancing");
}

void GasTransferSystem::SetSlaeRowNumsForVertices() {
PROFILE_FUNC();
//LOG4CPLUS_INFO(log_, "SetSlaeRowNumForVertices");
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
//LOG4CPLUS_INFO(log_, "End_SetSlaeRowNumForVertices");
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
PROFILE_FUNC();
//LOG4CPLUS_INFO(log_, "FormSlae");
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
  //LOG4CPLUS_INFO(log_, "EndFormSlae");
}
void GasTransferSystem::set_slae_solver(SlaeSolverI *slae_solver) {
PROFILE_FUNC();
//LOG4CPLUS_INFO(log_, "SetSlaeSolver");
  slae_solver_ = slae_solver;
}
void GasTransferSystem::set_manager_edge(ManagerEdge* manager_edge) {
PROFILE_FUNC();
//LOG4CPLUS_INFO(log_, "SetManagerEdge");
  g_->set_manager(manager_edge);
}
// Решить сформированную СЛАУ и найти вектор DeltaP_.
void GasTransferSystem::SolveSlae() {
PROFILE_FUNC();
//LOG4CPLUS_INFO(log_, "SolveSlae");
  // Переводим map A((row,col), val),  в список (row*col, val) для сортировки.
  int size = B_.size();
  std::list<std::pair<int, double> > flat;
  for(auto a = A_.begin(); a != A_.end(); ++a) {
    flat.push_back( 
        std::make_pair(
            a->first.first*size + a->first.second, // row*size + col.
            a->second // value.
        )
    );
  }
  flat.sort();
  // Заполняем a_indexes, a_vals на основании flat.
  std::vector<int> a_indexes;
  a_indexes.reserve( flat.size() );
  std::vector<double> a_vals;
  a_vals.reserve( flat.size() );
  for(auto fl = flat.begin(); fl != flat.end(); ++fl) {
    a_indexes.push_back(fl->first);
    a_vals.push_back(fl->second);
  }
  DeltaP_.clear();
  DeltaP_.reserve(size);
  slae_solver_->Solve( // Решаем СЛАУ, результат в вектор DeltaP_.
      a_indexes,
      a_vals,
      B_,
      &DeltaP_);
// Нужно было сгенерировать эталон разово и очень срочно.
//#define GEN_SARATOV_ETALON
#ifdef GEN_SARATOV_ETALON
  std::ofstream ofs("C:/Enisey/out/Testing/SLAE/Saratov_A_indices.txt");
  assert(ofs.good());
  boost::archive::text_oarchive oa(ofs);
  oa << a_indexes;

  std::ofstream ofs2("C:/Enisey/out/Testing/SLAE/Saratov_A_vals.txt");
  assert(ofs2.good());
  boost::archive::text_oarchive oa2(ofs2);
  oa2 << a_vals;

  std::ofstream ofs3("C:/Enisey/out/Testing/SLAE/Saratov_b.txt");
  assert(ofs3.good());
  boost::archive::text_oarchive oa3(ofs3);
  oa3 << B_;

  std::ofstream ofs4("C:/Enisey/out/Testing/SLAE/Saratov_x.txt");
  assert(ofs4.good());
  boost::archive::text_oarchive oa4(ofs4);
  oa4 << DeltaP_;
#endif
  //LOG4CPLUS_INFO(log_, "End_SolveSlae");
}

void GasTransferSystem::CountNewIteration(double g) {
PROFILE_FUNC();
//LOG4CPLUS_INFO(log_, "CountNewIteration");
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
  CopyGasStateFromVerticesToEdges();
  CountAllEdges();
  MixVertices();
//LOG4CPLUS_INFO(log_, "EndCountNewIteration");
}
double GasTransferSystem::CountDisbalance() {
PROFILE_FUNC();
//LOG4CPLUS_INFO(log_, "CountDisbalance");
  double d(0.0);
  for(auto v = g_->VertexBeginTopological(); v != g_->VertexEndTopological();
      ++v) {
      d += abs( v->CountDisbalance() );
  }
  return d;
//LOG4CPLUS_INFO(log_, "EndCountDisbalance");
}

int GasTransferSystem::GetIntDisbalance() {
PROFILE_FUNC();
//LOG4CPLUS_INFO(log_, "GetIntDisbalace");
  int s = 0;
  for(auto v = g_->VertexBeginTopological(); v != g_->VertexEndTopological();
    ++v) {
      if( v->AcceptableDisbalance(0.1) == false ) {
        ++s;
      }
  }
  return s;
//LOG4CPLUS_INFO(log_, "End_GetIntDisbalance");
}

void GasTransferSystem::LoadFromVestaFiles(std::string const path) {
PROFILE_FUNC();
//LOG4CPLUS_INFO(log_, "LoadFromVestaFiles");
  VestaFilesData vfd;
  std::ifstream mcf(path + "MatrixConnections.dat");
  LoadMatrixConnections(mcf, &vfd);

  std::ifstream plf(path + "PipeLine.dat");
  LoadPipeLines(plf, &vfd);

  std::ifstream iogf(path + "InOutGRS.dat");
  LoadInOutGRS(iogf, &vfd);

  GraphBoostLoadFromVesta(g_, &vfd);
//LOG4CPLUS_INFO(log_, "End_LoadFromVestaFiles");
}

void VectorOfStringToStream(const std::vector<std::string> &v, std::ofstream *o){
PROFILE_FUNC();
  for(auto l = v.begin(); l!= v.end(); ++l) {
    *o << *l;
    if( (l + 1) != v.end() ) {
      *o << std::endl;
    }
  }
}

void GasTransferSystem::LoadFromVestaFiles(
    const std::vector<std::string> &MatrixConnectionsFile,
    const std::vector<std::string> &InOutGRSFile,
    const std::vector<std::string> &PipeLineFile) {
PROFILE_FUNC();
//LOG4CPLUS_INFO(log_, "LoadFromVestaFiles");  
  VestaFilesData vsd;

  std::ofstream mco;
  mco.open("MatrixConnections.tmp", std::ios_base::trunc);
  VectorOfStringToStream(MatrixConnectionsFile, &mco);
  mco.close();
  std::ifstream mci("MatrixConnections.tmp");
  LoadMatrixConnections(mci, &vsd);

  std::ofstream plo;
  plo.open("PipeLine.tmp", std::ios_base::trunc);
  VectorOfStringToStream(PipeLineFile, &plo);
  plo.close();
  std::ifstream pli("PipeLine.tmp");
  LoadPipeLines(pli, &vsd);

  std::ofstream ioo;
  ioo.open("InOutGRS.tmp", std::ios_base::trunc);
  VectorOfStringToStream(InOutGRSFile, &ioo);
  ioo.close();
  std::ifstream ioi("InOutGRS.tmp");
  LoadInOutGRS(ioi, &vsd);

  GraphBoostLoadFromVesta(g_, &vsd);
//LOG4CPLUS_INFO(log_, "End_LoadFromVestaFiles");
}

void const GasTransferSystem::WriteToGraphviz(std::string const filename) {
PROFILE_FUNC();
  //LOG4CPLUS_INFO(log_, "WriteToGraphviz");
  WriterGraphviz writer;
  writer.WriteGraphToFile(*g_, filename);
//LOG4CPLUS_INFO(log_, "End_WriteToGraphviz");
}

void GasTransferSystem::MakeInitialApprox() {
PROFILE_FUNC();
  //LOG4CPLUS_INFO(log_, "MakeInitialApprox");
  double overall_p_min(999.0);
  double overall_p_max(-999.0);
  //LOG4CPLUS_INFO(log_, "BeforeFindOverallMinAndMax");
  FindOverallMinAndMaxPressureConstraints(
      g_, 
      &overall_p_max,
      &overall_p_min);
  // 1. Рассчитываем overall ограничения по всему графу.
  //LOG4CPLUS_INFO(log_, "BeforeSetPressureConstraints");
  //try {
    SetPressureConstraintsForVertices(
        g_,
        overall_p_min,
        overall_p_max );
  //} catch(std::exception ex) {LOG4CPLUS_INFO(log_, ex.what());}
  // 3. Задаём начальное приближение давлений.
  //LOG4CPLUS_INFO(log_, "BeforeSetInitialApproxPressures");
  SetInitialApproxPressures(g_, overall_p_max, overall_p_min);
  // 4. Задаём начальное приближение температур.
  //LOG4CPLUS_INFO(log_, "BeforeSetInitialApproxTemperatures");
  SetInitialApproxTemperatures(g_, 278.0);
  // 5. Протягиваем состав газа от входов к выходам.
  //LOG4CPLUS_INFO(log_, "BeforeMakeInitialMix");
  MakeInitialMix();
//LOG4CPLUS_INFO(log_, "End_MakeInitialApprox");
}

void GasTransferSystem::CopyGasStateFromVerticesToEdges() {
PROFILE_FUNC();
//LOG4CPLUS_INFO(log_, "CopyGasStateFromVerticesToEdges");
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
//LOG4CPLUS_INFO(log_, "CopyGasStateFromVerticesToEdges");
}

// Внимание! Перед вызовом этого метода необходимо вызывать 
// метод GasTransferSystem::CopyGasStateFromVerticesToEdges().
// Расцепил их, чтобы была возможность сформировать эталонные работчи параметры
// После передачи параметров газового потока в рёбра, но до их моделирования.
void GasTransferSystem::CountAllEdges() {  
PROFILE_FUNC();
//LOG4CPLUS_INFO(log_, "CountAllEdges");
  g_->manager()->CountAll();
//LOG4CPLUS_INFO(log_, "EndCountAllEdges");
}

void GasTransferSystem::MakeInitialMix() {
PROFILE_FUNC();
//LOG4CPLUS_INFO(log_, "MakeInitialMix");
  for(auto v = g_->VertexBeginTopological(); v != g_->VertexEndTopological();
    ++v) {
    v->InitialMix();
  }
//LOG4CPLUS_INFO(log_, "End_MakeInitialMix");
}

void GasTransferSystem::MixVertices() {
PROFILE_FUNC();
//LOG4CPLUS_INFO(log_, "MixVertices");
  for(auto v = g_->VertexBeginTopological(); v != g_->VertexEndTopological();
      ++v) {
    v->MixGasFlowsFromAdjacentEdges();
  }
//LOG4CPLUS_INFO(log_, "End_MixVertices");
}

ManagerEdgeModelPipeSequential* GasTransferSystem::
    manager_model_pipe_sequential() {
  return static_cast<ManagerEdgeModelPipeSequential*>( g_->manager() );
}
