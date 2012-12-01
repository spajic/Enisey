/** \file SleSolverCvmService.h
IceBox сервис решения СЛАУ на базе CVM.
Реализаует интерфейс SlaeSolverIceCVM - наследние SlaeSolverIce. */

#pragma once 

#include <IceBox/IceBox.h>

class SlaeSolverCvmSerivce : public ::IceBox::Service {
public:
  SlaeSolverCvmSerivce();
  virtual ~SlaeSolverCvmSerivce();

  virtual void start(
      const ::std::string&,
      const ::Ice::CommunicatorPtr&,
      const ::Ice::StringSeq&);

  virtual void stop();

private:
  ::Ice::ObjectAdapterPtr adapter_;
};
