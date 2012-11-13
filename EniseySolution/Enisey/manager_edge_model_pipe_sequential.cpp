#include "manager_edge_model_pipe_sequential.h"

#include <fstream>

#include <algorithm>

#include <ppl.h>

#include "edge_model_pipe_sequential.h"
#include "passport_pipe.h"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

#include "work_params.h"
#include "calculated_params.h"


ManagerEdgeModelPipeSequential::ManagerEdgeModelPipeSequential()
{
  models_.reserve(128);
  edges_.reserve(128);
  /// \todo Взять размер вектора исходя из кол-во объектов ГТС.
  max_index_ = 0;
}

Edge* ManagerEdgeModelPipeSequential::CreateEdge(const Passport* passport)
{
  models_.push_back( ModelPipeSequential(passport) );

  EdgeModelPipeSequential edge;
  edge.pointer_to_model_ = &(models_[max_index_]);
  edges_.push_back(edge);
  ++max_index_;
  return &(edges_[max_index_ - 1]);
}

void ManagerEdgeModelPipeSequential::CountAll()
{
  Concurrency::parallel_for_each(models_.begin(), models_.end(), [](ModelPipeSequential &model)
    //std::for_each(models_.begin(), models_.end(), [](ModelPipeSequential &model)
  {
    model.Count();
  } );
}

void ManagerEdgeModelPipeSequential::FinishAddingEdges()
{

}

template<class VectorElement>
void SaveVectorToFile( std::string file_name, std::vector<VectorElement> vec ) {
  std::ofstream ofs(file_name);
  assert(ofs.good());
  boost::archive::xml_oarchive oa(ofs);
  oa << BOOST_SERIALIZATION_NVP(vec);
  // archive and stream closed when destructors are called
}

void ManagerEdgeModelPipeSequential::
    SavePassportsToFile(std::string file_name) {  
  std::vector<PassportPipe> passports;
  for(auto m = models_.begin(); m != models_.end(); ++m) {
    // Сейчас вектор models_ имеет зарезервированный размер, чтобы положение
    // объектов в памяти не менялось и можно было записывать в граф указатели
    // на них. Это плохо, надо исправить.
    // Проверка length > 0 позволяет не добавлять трубы, место для которых
    // зарезервировано, но на самом деле их в расчётной схеме нет.
    if(m->passport()->length_ > 0) {
      passports.push_back( *(m->passport()) );      
    }
  }
  SaveVectorToFile(file_name, passports);
}

// Внимание! Мы передаём менеджеру параллельных расчётов объекты всегде
// в не реверсивном виде. Об этом должен заботиться суперменеджер.
// Поэтому здесь всегда всё в естественном порядке, реверсивных труб нет.
void ManagerEdgeModelPipeSequential::
    SaveWorkParamsToFile(std::string file_name) {
  std::vector<WorkParams> work_params;  
  for(auto m = models_.begin(); m != models_.end(); ++m) {
    WorkParams wp;       
    Gas gas_in  = m->gas_in();
    Gas gas_out = m->gas_out();        
    wp.set_den_sc_in  (gas_in.  composition.density_std_cond);
    wp.set_co2_in     (gas_in.  composition.co2);
    wp.set_n2_in      (gas_in.  composition.n2);
    wp.set_p_in       (gas_in.  work_parameters.p);
    wp.set_t_in       (gas_in.  work_parameters.t);
    wp.set_p_out      (gas_out. work_parameters.p);
    work_params.push_back( wp );
  } // Конец цикла по всем моделям.
  SaveVectorToFile(file_name, work_params);
}

void ManagerEdgeModelPipeSequential::
    SaveCalculatedParamsToFile(std::string file_name) {
  std::vector<CalculatedParams> calculated_params;  
  for(auto m = models_.begin(); m != models_.end(); ++m) {    
    CalculatedParams cp;
    cp.set_q        ( m->q() );    
    cp.set_dq_dp_in ( m->dq_dp_in() );
    cp.set_dq_dp_out( m->dq_dp_out() );
    cp.set_t_out    ( m->gas_out().work_parameters.t);
    calculated_params.push_back(cp);
  } // Конец цикла по всем моделям.
  SaveVectorToFile(file_name, calculated_params);
}


