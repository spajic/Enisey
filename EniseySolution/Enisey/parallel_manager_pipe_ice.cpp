/** \file parallel_manager_pipe_ice.cpp
Менеджер параллельного моделирования труб, использующий сервис ICE.*/
#pragma once
#include "parallel_manager_pipe_ice.h"
#include "ParallelManagerIceI.h"

ParallelManagerPipeIce::ParallelManagerPipeIce() {
  try {
    ic = Ice::initialize();
    Ice::ObjectPrx base = 
        ic->stringToProxy("ParallelManagerIce:default -p 10000");
    proxy = Enisey::ParallelManagerIceIPrx::checkedCast(base);
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

void ParallelManagerPipeIce::
    TakeUnderControl(std::vector<PassportPipe> const &passports) {
  //proxy->TakeUnderControl();
}
void ParallelManagerPipeIce::
    SetWorkParams(std::vector<WorkParams> const &work_params) {
  //parallel_manager_proxy->SetWorkParams();
}
void ParallelManagerPipeIce::CalculateAll() {
  proxy->CalculateAll();
}
void ParallelManagerPipeIce::GetCalculatedParams(
    std::vector<CalculatedParams> *calculated_params) {
  //proxy->GetCalculatedParams();
}