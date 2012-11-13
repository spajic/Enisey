#pragma once
#include "passport.h"

#include <string>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>

// Наследуем структуру PassportPipe от пустого интерфейсного класса Passport
// для единообразия. Все паспортные параметры, необходимые для создания объктов
// собираются в структуры и наследуются от Passport.h
// Планируется сделать интерфейс Менеджера рёбер, получающий для создания
// ребра как раз объект типа Passport.
struct PassportPipe: public Passport
{
public:
  PassportPipe();
  std::string GetName(); 

  double length_;						// Длина [км]
  double d_outer_;						// Внешний диаметр [мм]
  double d_inner_;						// Внутренний диаметр [мм]
  double p_max_;						// Макс. допустимое давл-е [МПа]
  double p_min_;						// Мин. допустимое давл-е [МПа]
  double hydraulic_efficiency_coeff_;	// Коэф-т гидавлич эф-ти [б.р.]
  double roughness_coeff_;				// Коэф-т эквив. шерхов-ти [б.р.]
  // Включаем в пасспорт температур о.с. потому что она задаётся при создании
  // трубы и далее не меняется (по-крайней мере пока - в контексте рассмотр.
  // задач)
  double heat_exchange_coeff_;			// К-т теплообм с окр. ср.[Вт / (м2*К)]
  double t_env_;						// Темп. окр. среды [К]
private:
  // Для сериализации.
  friend class boost::serialization::access;
  template<class Archive> 
  void serialize(Archive & ar, const unsigned int version) {
    ar & BOOST_SERIALIZATION_NVP(d_inner_);
    ar & BOOST_SERIALIZATION_NVP(d_outer_);
    ar & BOOST_SERIALIZATION_NVP(heat_exchange_coeff_);
    ar & BOOST_SERIALIZATION_NVP(hydraulic_efficiency_coeff_);
    ar & BOOST_SERIALIZATION_NVP(length_);
    ar & BOOST_SERIALIZATION_NVP(p_max_);
    ar & BOOST_SERIALIZATION_NVP(p_min_);
    ar & BOOST_SERIALIZATION_NVP(roughness_coeff_);
    ar & BOOST_SERIALIZATION_NVP(t_env_);
  }
};