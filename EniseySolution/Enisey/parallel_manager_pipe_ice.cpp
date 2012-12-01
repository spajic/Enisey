/** \file parallel_manager_pipe_ice.cpp
Менеджер параллельного моделирования труб, использующий сервис ICE.*/
#pragma once
#include "parallel_manager_pipe_ice.h"
#include "ParallelManagerIce.h"

ParallelManagerPipeIce::ParallelManagerPipeIce() {
  try {
    Ice::PropertiesPtr props = Ice::createProperties();
    // Make sure that network and protocol tracing are off.
    //
    props->setProperty("Ice.MessageSizeMax", "10240");
    // Initialize a communicator with these properties.
    //
    Ice::InitializationData id;
    id.properties = props;
    ic = Ice::initialize(id);    
    Ice::ObjectPrx base = 
        ic->stringToProxy("ParallelManagerIce:tcp -h 127.0.0.1 -p 10000");
    proxy = Enisey::ParallelManagerIcePrx::checkedCast(base);
    if(!proxy) {
      std::cout << "Invalid proxy ParallelManagerPipeIce" << std::cout;
    };
  }
  catch(const Ice::Exception &ex){  
    std::cerr << ex << std::endl;
  }
}
ParallelManagerPipeIce::ParallelManagerPipeIce(std::string endpoint) {
  try {
    Ice::PropertiesPtr props = Ice::createProperties();
    // Make sure that network and protocol tracing are off.
    //
    props->setProperty("Ice.MessageSizeMax", "10240");
    // Initialize a communicator with these properties.
    //
    Ice::InitializationData id;
    id.properties = props;
    ic = Ice::initialize(id); 
    Ice::ObjectPrx base = 
      ic->stringToProxy(endpoint);
    proxy = Enisey::ParallelManagerIcePrx::checkedCast(base);
    if(!proxy) {
      std::cout << "Invalid proxy ParallelManagerPipeIce" << std::cout;
    };
  }
  catch(const Ice::Exception &ex){  
    std::cerr << ex << std::endl;
  }
}
ParallelManagerPipeIce::~ParallelManagerPipeIce() {
  if(ic) {
    try {
      ic->destroy();
    } catch(const Ice::Exception &e) {
      std::cerr << e << std::endl;
    }
  }
}

void ParallelManagerPipeIce::SetParallelManagerType(std::string type) {
  if(type == "CUDA") {
    proxy->SetParallelManagerType(Enisey::CUDA);
  }
  else if(type == "OpenMP") {
    proxy->SetParallelManagerType(Enisey::OpenMP);
  }
  else if(type == "SingleCore") {
    proxy->SetParallelManagerType(Enisey::SingleCore);
  }
}

void ParallelManagerPipeIce::
    TakeUnderControl(std::vector<PassportPipe> const &passports) {      
  Enisey::PassportSequence p_seq;
  ConvertPassportVecToSequence(passports, &p_seq);
  proxy->TakeUnderControl(p_seq);
}

void ParallelManagerPipeIce::
    SetWorkParams(std::vector<WorkParams> const &work_params) {
  Enisey::WorkParamsSequence wp_seq;
  ConvertWorkParamsVecToSequence(work_params, &wp_seq);
  proxy->SetWorkParams(wp_seq);
}

void ParallelManagerPipeIce::CalculateAll() {
  proxy->CalculateAll();
}

void ParallelManagerPipeIce::GetCalculatedParams(
    std::vector<CalculatedParams> *calculated_params) {
  Enisey::CalculatedParamsSequence cp_seq;
  proxy->GetCalculatedParams(cp_seq);
  ConvertCalculatedParamsSeqToVector(cp_seq, calculated_params);
}

// ---Функции конвертации векторов в последовательности------------------------
void ParallelManagerPipeIce::ConvertPassportVecToSequence( 
    std::vector<PassportPipe> const &passports,
    Enisey::PassportSequence *passport_seq) {
  passport_seq->reserve( passports.size() );
  for(auto it = passports.begin(); it != passports.end(); ++it) {
    Enisey::PassportPipeIce pipe_ice;
    pipe_ice.dInner     = it->d_inner_;
    pipe_ice.dOuter     = it->d_outer_;
    pipe_ice.heatExch   = it->heat_exchange_coeff_;
    pipe_ice.hydrEff    = it->hydraulic_efficiency_coeff_;
    pipe_ice.length     = it->length_;
    pipe_ice.pMax       = it->p_max_;
    pipe_ice.pMin       = it->p_min_;
    pipe_ice.roughCoeff = it->roughness_coeff_;
    pipe_ice.tEnv       = it->t_env_;
    passport_seq->push_back(pipe_ice);
  } 
}

void ParallelManagerPipeIce::ConvertCalculatedParamsSeqToVector( 
    Enisey::CalculatedParamsSequence &cp_seq,
    std::vector<CalculatedParams> *calculated_params ) {
  calculated_params->clear();
  calculated_params->reserve(cp_seq.size());
  for(auto it = cp_seq.begin(); it != cp_seq.end(); ++it) {
    CalculatedParams cp;
    cp.set_q         (it->q);
    cp.set_dq_dp_in  (it->dqDpIn);
    cp.set_dq_dp_out (it->dqDpOut);
    cp.set_t_out     (it->tOut);
    calculated_params->push_back(cp);
  }
}

void ParallelManagerPipeIce::ConvertWorkParamsVecToSequence(
    std::vector<WorkParams> const &work_params,
    Enisey::WorkParamsSequence *wp_seq) {
  wp_seq->reserve( work_params.size() );
  for(auto it = work_params.begin(); it != work_params.end(); ++it) {
    Enisey::WorkParamsIce wp_ice;
    wp_ice.co2In    = it->co2_in();
    wp_ice.denScIn  = it->den_sc_in();
    wp_ice.n2In     = it->n2_in();
    wp_ice.pIn      = it->p_in();
    wp_ice.pOut     = it->p_out();
    wp_ice.tIn      = it->t_in();
    wp_seq->push_back(wp_ice);
  }
}