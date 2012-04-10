/** \file loader_vesta.cpp
Реализация функционала, объявленного и прокомментированного в соответствующем
заголовочном файле.
*/
#include "loader_vesta.h"
#include <string>
#include <hash_map>
#include <fstream>
#include "passport_pipe.h"

// Реализации конструкторов по умолчанию для структур из loader_vesta.h
EdgeData::EdgeData() : id_vesta(-1), in_vertex_id(-1), out_vertex_id(-1), 
    edge_type(-1),  pipe_type(-1) {}
InOutData::InOutData() : id(-1), q(-1), temp(-1), den_sc(-1), pressure(-1), 
    n2(-1),  co2(-1), heat_making(-1), id_in_out(-1), min_p(-1) {}
VertexData::VertexData() : id_vesta(-1), id_graph(-1) {}
VestaFilesData::VestaFilesData() : num_edges(-1), num_nodes(-1), 
    elems_per_string(-1) {}

void LoadMatrixConnections(std::string file_name, VestaFilesData *vsd) {
  std::fstream f(file_name);
  f >> vsd->num_edges >> vsd->num_nodes >> vsd->elems_per_string;
  // Считываем информацию об edges.
  for(int i = 0; i < vsd->num_edges; ++i) {
    EdgeData edge;
    f >> edge.id_vesta >> edge.edge_type;
    vsd->edges_hash[edge.id_vesta] = edge;
  }
  // Считываем информацию о nodes.
  for(int i = 0; i < vsd->num_nodes; ++i) {
    VertexData vertex;
    f >> vertex.id_vesta;
    // Считываем информацию о входящих и исходящих в данный узел рёбрах.
    for(int inout_num = 0; inout_num < vsd->elems_per_string; ++inout_num) {
      int in_out_edge_id = -1;
      f >> in_out_edge_id;
      if(in_out_edge_id == 0) { } // Остаточные нули.
      if(in_out_edge_id < 0) { // Входящее ребро.
        vertex.edges_id_in.push_back(-in_out_edge_id);
        // Ставим входящему в узел ребру этот узел как out_vertex.
        vsd->edges_hash[-in_out_edge_id].out_vertex_id = vertex.id_vesta;
      }
      if(in_out_edge_id > 0) { // Исходящее ребро.
        vertex.edges_id_out.push_back(in_out_edge_id);
        // Ставим исходящему из узла ребру этот узел как in_vertex.
        vsd->edges_hash[in_out_edge_id].in_vertex_id = vertex.id_vesta;
      }
      // Запоминаем заполненный vertex в хеше vsd.
      vsd->vertices_hash[vertex.id_vesta] = vertex;
    }
  }
}
void LoadPipeLines(std::string file_name, VestaFilesData *vsd) {
  std::fstream f(file_name);
  int total_pipes = -1;
  f >> total_pipes;
  for(int pipe_num = 1; pipe_num <= total_pipes; ++pipe_num) {
    PassportPipe passport;
    /** \todo Переменные, заполняемые здесь не идут никуда дальше. 
    * разобраться с этим */
    int pipe_id = -1;
    int pipe_type = -1;
    int pipe_mg = -1; // Номер нитки МГ.
    double pipe_height_difference = 0;
    f >> pipe_id >> pipe_type >> pipe_mg;
    f >> passport.hydraulic_efficiency_coeff_ >> passport.heat_exchange_coeff_;
    f >> passport.t_env_; 
    passport.t_env_ += 273.15; // Перевод из [C] в [K].
    f >> passport.p_min_ >> passport.p_max_;
    passport.p_max_ *= 0.0980665; // Перевод из [Ата] в [Мпа].
    passport.p_min_ *= 0.0980665;
    f >> passport.d_inner_ >> passport.d_outer_ >> passport.length_;
    f >> passport.roughness_coeff_ >> pipe_height_difference;
    vsd->edges_hash[pipe_id].passport = passport;
    vsd->edges_hash[pipe_id].pipe_type = pipe_type;
  }  
}

void LoadInOutGRS(std::string file_name, VestaFilesData *vsd) {
  std::fstream f(file_name);
  int total_inouts = -1;
  f >> total_inouts;
  for(int num_inout = 1; num_inout <= total_inouts; ++num_inout) {
    InOutData inout;
    /** \todo Переменные, заполняемые здесь не идут никуда дальше. 
    * разобраться с этим */
    int control_pressure = -1; // Признак контроля давления.
    int id_pipeline = -1; // Идентификатор нитки.
    double humidity = -1; // Влажность газа.
    double Pst = -1; // Новые параметры Pst, A, B, Qmax.
    double A = -1; 
    double B = -1;
    double Qmax = -1; 
    f >> inout.id >> inout.q;
    inout.q /= 0.0864; // Перевод из [млн м3/cут] в [м3/cек].
    f >> inout.temp; 
    inout.temp += 273.15; // Перевод из [C] в [K].
    f >> inout.den_sc;
    f >> inout.pressure;
    inout.pressure *= 0.0980665; // Перевод из [Ата] в [МПа].
    f >> inout.n2 >> inout.co2 >> inout.heat_making;
    f >> control_pressure;
    f >> inout.id_in_out >> inout.min_p;
    inout.min_p *= 0.0980665; // Перевод из [Ата] в [МПа].
    f >> id_pipeline;
    f >> humidity;
    f >> Pst >> A >> B >> Qmax;
    vsd->vertices_hash[inout.id].in_outs.push_back(inout);
  }
}