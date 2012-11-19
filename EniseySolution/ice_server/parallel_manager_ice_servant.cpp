/** \file parallel_manager_ice.cpp
Класс ParallelManagerIce - реалиазация интерфейса ParallelManagerIceI*/
#include "parallel_manager_ice_servant.h"
#include <Ice/Ice.h>
#include <iostream>
#include "passport_pipe.h"
#include "work_params.h"
#include "calculated_params.h"
#include "parallel_manager_pipe_singlecore.h"
#include "parallel_manager_pipe_openmp.h"
#include "parallel_manager_pipe_cuda.cuh"

#include <log4cplus/logger.h>
#include <log4cplus/loggingmacros.h>

using namespace log4cplus;

namespace Enisey {
ParallelManagerIceServant::ParallelManagerIceServant() {
  manager_ = new ParallelManagerPipeCUDA;
  log = Logger::getInstance( LOG4CPLUS_TEXT("IceServer.PMServant") );
}

ParallelManagerIceServant::~ParallelManagerIceServant() {
  delete manager_;
}

void ParallelManagerIceServant::
    SetParallelManagerType(ParallelManagerType type, const ::Ice::Current&) {  
LOG4CPLUS_INFO(log, "Start SetParallelManagerType");
  // Удаляем предыдущий менеджер - он всегда есть, так как впервые задаётся
  // в конструкторе, а затем пересоздаётся здесь.
  delete manager_; 
  switch(type) {
  case ParallelManagerType::SingleCore :
LOG4CPLUS_INFO(log, "--Set type = SingleCore");
    manager_ = new ParallelManagerPipeSingleCore;
    break;
  case ParallelManagerType::OpenMP :
LOG4CPLUS_INFO(log, "--Set type = OpenMP");
    manager_ = new ParallelManagerPipeOpenMP;
    break;
  case ParallelManagerType::CUDA :
LOG4CPLUS_INFO(log, "--Set type = CUDA");
    manager_ = new ParallelManagerPipeCUDA;
    break;
  }    
LOG4CPLUS_INFO(log, "End SetParallelManagerType");
}

void ParallelManagerIceServant::TakeUnderControl( 
    const ::Enisey::PassportSequence& passport_seq, 
    const ::Ice::Current& ) {
LOG4CPLUS_INFO(log, "Start TakeUnderControl");
LOG4CPLUS_INFO(log, "--Size: " << passport_seq.size() );
  SavePassportSequenceToVector(passport_seq);
  manager_->TakeUnderControl(passports_);
LOG4CPLUS_INFO(log, "End TakeUnderControl");
}

void ParallelManagerIceServant::SetWorkParams(
    const ::Enisey::WorkParamsSequence& wp_seq, 
    const ::Ice::Current&) {  
LOG4CPLUS_INFO(log, "Start SetWorkParams");
  SaveWorkParamsSequenceToVector(wp_seq);  
  manager_->SetWorkParams(work_params_);
LOG4CPLUS_INFO(log, "End SetWorkParams");
}

void ParallelManagerIceServant::CalculateAll(const ::Ice::Current&) {
LOG4CPLUS_INFO(log, "Start CalculateAll");
  manager_->CalculateAll();
LOG4CPLUS_INFO(log, "End CalculateAll");
}

void ParallelManagerIceServant::GetCalculatedParams(
  ::Enisey::CalculatedParamsSequence& cp_seq, 
  const ::Ice::Current&) {
LOG4CPLUS_INFO(log, "Start GetCalculatedParams");  
  manager_->GetCalculatedParams(&calculated_params_);
  SaveCalculatedParamsVectorToSequence(cp_seq);
LOG4CPLUS_INFO(log, "End GetCalculatedParams");  
}

void ParallelManagerIceServant::
    ActivateSelfInAdapter(const Ice::ObjectAdapterPtr &adapter) {
  Ice::Identity id;
  id.name = "ParallelManagerIce";
  try {
    adapter->add(this, id);
  } catch(const Ice::Exception &ex) {
    std::cout << ex.what();
  }
}
//-------Функции преобразования параметров между ICE и C++---------------------
void ParallelManagerIceServant::
    SavePassportSequenceToVector(
        const ::Enisey::PassportSequence &passport_seq) {
  passports_.clear();
  passports_.reserve( passport_seq.size() );
  for(auto p_it = passport_seq.begin(); p_it != passport_seq.end(); ++p_it) {
    PassportPipe p;
    p.d_inner_ = p_it->dInner;
    p.d_outer_ = p_it->dOuter;
    p.heat_exchange_coeff_ = p_it->heatExch;
    p.hydraulic_efficiency_coeff_ = p_it->hydrEff;
    p.length_ = p_it->length;
    p.p_max_ = p_it->pMax;
    p.p_min_ = p_it->pMin;
    p.roughness_coeff_ = p_it->roughCoeff;
    p.t_env_ = p_it->tEnv;
    passports_.push_back(p);
  }
}

void ParallelManagerIceServant::
    SaveWorkParamsSequenceToVector(const ::Enisey::WorkParamsSequence &wp_seq){
  work_params_.clear();
  work_params_.reserve( wp_seq.size() );
  for(auto wp_it = wp_seq.begin(); wp_it != wp_seq.end(); ++wp_it) {
    WorkParams wp;
    wp.set_co2_in   (wp_it->co2In   );
    wp.set_den_sc_in(wp_it->denScIn );
    wp.set_n2_in    (wp_it->n2In    );
    wp.set_p_in     (wp_it->pIn     );
    wp.set_p_out    (wp_it->pOut    );
    wp.set_t_in     (wp_it->tIn     );
    work_params_.push_back(wp);
  }
}

void ParallelManagerIceServant::
    SaveCalculatedParamsVectorToSequence(
        ::Enisey::CalculatedParamsSequence &cp_seq ) {
  cp_seq.clear();
  cp_seq.reserve(calculated_params_.size());
  for(auto it = calculated_params_.begin(); it != calculated_params_.end(); 
    ++it) {
      CalculatedParamsIce cp_ice;
      cp_ice.q        = it->q();
      cp_ice.dqDpIn   = it->dq_dp_in();
      cp_ice.dqDpOut  = it->dq_dp_out();
      cp_ice.tOut     = it->t_out();    
      cp_seq.push_back(cp_ice);
  }
}

} // Конец namespace Enisey.