/** \file work_params.h
Класс WorkParams - рабочие параметры моделируемого объекта. */
#pragma once
#include <boost/serialization/access.hpp>
class WorkParams {
public:
  WorkParams();
  WorkParams(
    double den_sc_in, 
    double co2_in, 
    double n2_in, 
    double p_in, 
    double t_in, 
    double p_out);  
  void set_den_sc_in(double den_sc_in);
  void set_co2_in(double co2_in);
  void set_n2_in(double n2_in);
  void set_p_in(double p_in);
  void set_t_in(double t_in);
  void set_p_out(double p_out);
  double den_sc_in();
  double co2_in();
  double n2_in();
  double p_in();
  double t_in();
  double p_out();
private:
  double den_sc_in_; ///< Плотность при стандартных условиях на входе.
  double co2_in_; ///< Содержание углекислого газа на входе.
  double n2_in_; ///< Содержание азота на входе.
  double p_in_; ///< Давление на входе.
  double t_in_; ///< Темпетарута на входе.
  double p_out_; ///< Давление на выходе.
  // Для сериализации.
  friend class boost::serialization::access;
  template<class Archive> 
      void serialize(Archive & ar, const unsigned int version) {
        ar & BOOST_SERIALIZATION_NVP( p_in_ );
        ar & BOOST_SERIALIZATION_NVP( p_out_ );
        ar & BOOST_SERIALIZATION_NVP( t_in_ );
        ar & BOOST_SERIALIZATION_NVP( den_sc_in_ );
        ar & BOOST_SERIALIZATION_NVP( co2_in_ );
        ar & BOOST_SERIALIZATION_NVP( n2_in_ );
      }
};