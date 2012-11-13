/** \file work_params.cpp
Реализация WorkParams - рабочие параметры моделируемого объекта. */
#include "work_params.h"
WorkParams::WorkParams() : 
    den_sc_in_  (-1), 
    co2_in_     (-1), 
    n2_in_      (-1), 
    p_in_       (-1), 
    t_in_       (-1),
    p_out_      (-1) {};
WorkParams::WorkParams(
  double den_sc_in, 
  double co2_in, 
  double n2_in, 
  double p_in, 
  double t_in, 
  double p_out) :
    den_sc_in_  (den_sc_in),
    co2_in_     (co2_in),
    n2_in_      (n2_in),
    p_in_       (p_in),
    t_in_       (t_in),
    p_out_      (p_out) {};
void WorkParams::set_den_sc_in(double den_sc_in)  { den_sc_in_ = den_sc_in; }
void WorkParams::set_co2_in   (double co2_in)     { co2_in_ = co2_in; }
void WorkParams::set_n2_in    (double n2_in)      { n2_in_ = n2_in; }
void WorkParams::set_p_in     (double p_in)       { p_in_ = p_in; }
void WorkParams::set_t_in     (double t_in)       { t_in_ = t_in; }
void WorkParams::set_p_out    (double p_out)      { p_out_ = p_out; }
double WorkParams::den_sc_in(){ return den_sc_in_; }
double WorkParams::co2_in()   { return co2_in_; }
double WorkParams::n2_in()    { return n2_in_; }
double WorkParams::p_in()     { return p_in_; }
double WorkParams::t_in()     { return t_in_; }
double WorkParams::p_out()    { return p_out_; }