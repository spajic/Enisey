/** \file SleSolverCvmService.cpp
IceBox сервис решения СЛАУ на базе CVM.
Реализаует интерфейс SlaeSolverIceCVM - наследние SlaeSolverIce. */

#include "slae_solver_cvm_service.h"
#include <Ice/Ice.h>
#include <IceBox/IceBox.h>

using namespace std;

extern "C" {
  ICE_DECLSPEC_EXPORT IceBox::Service* create(
      Ice::CommunicatorPtr communicator) {
          return new SlaeSolverCvmSerivce; 
  }
}

SlaeSolverCvmSerivce::SlaeSolverCvmSerivce() {
  std::cout << "Hello from SlaeSolverCvmService constructor!\n";
}

SlaeSolverCvmSerivce::~SlaeSolverCvmSerivce() {
  std::cout << "Goodbye from SlaeSolverCvmService destructor!\n";
}

void SlaeSolverCvmSerivce::start(
    const string                &name, 
    const Ice::CommunicatorPtr  &communicator, 
    const Ice::StringSeq        &args) {
  std::cout << "Start SlaeSolverCvmService\n";
  adapter_ = communicator->createObjectAdapter(name);
  //Demo::HelloPtr hello = new HelloI;
  //_adapter->add(hello, communicator->stringToIdentity("hello"));
  adapter_->activate();
}

void SlaeSolverCvmSerivce::stop() {
  std::cout << "Stop SlaeSolverCvmService\n";
  adapter_->destroy();
}