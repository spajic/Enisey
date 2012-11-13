/** \file calculated_params.h
Класс CalculatedParams - рассчитанные параметры моделируемого объекта. */
#pragma once
#include <boost/serialization/access.hpp>
class CalculatedParams {
public:
  CalculatedParams();
  CalculatedParams(
      double q, 
      double dq_dp_in, 
      double dq_dp_out, 
      double t_out);
  void set_q(double q);
  void set_dq_dp_in(double dq_dp_in);
  void set_dq_dp_out(double dq_dp_out);
  void set_t_out(double t_out);
  
  double q();
  double dq_dp_in();
  double dq_dp_out();
  double t_out();
private:
  double q_;          ///< Расход по объекту.
  double dq_dp_in_;   ///< Производная q по p_in;
  double dq_dp_out_;  ///< Производная q по p_out;
  double t_out_;      ///< Температура на выходе объекта.
  // Сериализация.
  // Для сериализации.
  friend class boost::serialization::access;
  template<class Archive> 
  void serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP( q_ );
    ar & BOOST_SERIALIZATION_NVP( dq_dp_in_ );
    ar & BOOST_SERIALIZATION_NVP( dq_dp_out_ );
    ar & BOOST_SERIALIZATION_NVP( t_out_ );    
  }
};