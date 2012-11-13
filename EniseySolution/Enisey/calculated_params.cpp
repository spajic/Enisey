/** \file calculated_params.cpp
Реализация CalculatedParams - раcсчитанные параметры моделируемого объекта. */
#include "calculated_params.h"
CalculatedParams::CalculatedParams() : 
    q_(-1), 
    dq_dp_in_(-1),
    dq_dp_out_(-1),
    t_out_(-1) {};
CalculatedParams::CalculatedParams(
  double q,
  double dq_dp_in,
  double dq_dp_out,
  double t_out) :
    q_(q),
    dq_dp_in_(dq_dp_in),
    dq_dp_out_(dq_dp_out),
    t_out_(t_out) {};
void CalculatedParams::set_q(double q)                { q_ = q; }
void CalculatedParams::set_dq_dp_in(double dq_dp_in)  { dq_dp_in_ = dq_dp_in; }
void CalculatedParams::set_dq_dp_out(double dq_dp_out){ dq_dp_out_ = dq_dp_out; }
void CalculatedParams::set_t_out(double t_out)        { t_out_ = t_out; }

double CalculatedParams::q()        { return q_; }
double CalculatedParams::dq_dp_in() { return dq_dp_in_; }
double CalculatedParams::dq_dp_out(){ return dq_dp_out_; }
double CalculatedParams::t_out()    { return t_out_; }