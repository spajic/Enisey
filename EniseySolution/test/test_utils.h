/** \file test_utils.h
Утилитарные функции для тестирования. */
#pragma once
#include <vector>
#include <iostream>

#include <boost/serialization/vector.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>


/// Константы для паспорта трубы.
/// Внутренний диаметр трубы, [мм].
const double kInnerPipeDiameter = 1000.0; 
/// Наружный диаметр трубы, [мм].
const double kOuterPipeDiameter = 1020.0;
/// Длина трубы, [км].
const double kPipeLength = 100.0;
/// Коэффициент теплообмена трубы с внешней средой, [].
const double kHeatExchangeCoefficient = 1.3;
/// Коэффициент гидравлической эффективности, [].
const double kHydraulicEfficiencyCoefficient = 0.95;
/// Ограничение на максимальное давление [МПа].
const double kMaximumPressure = 100.0;
/// Ограничение на минимальное давление [МПа].
const double kMinimumPressure = 1.0;
/// Коэффициент эквивалентной шероховатости [].
const double kRoughnessCoefficient = 0.03;
/// Температура окружающей среды [К].
const double kEnvironmentTemperature = 280.15;

/// Константы для входящего газового потока.
/// Плотность при стандартных условиях, [кг/м3].
const double kInputGasDensityOnStandartConditions = 0.6865365;
/// Содержание углекислого газа, [б.р.].
const double kInputGasCarbonDioxidePart = 0.0;
/// Содержание азота, [б.р.].
const double kInputGasNitrogenPart = 0.0;
/// Давление, [МПа].
const double kInputGasPressure = 5.0;
/// Температура, [К].
const double kInputGasTemperature = 293.15;
/// Расход, [м3/сек].
const double kInputGasQuantity = 387.843655734;

/// Константы для исходящего газового потока.
/// Плотность при стандартных условиях, [кг/м3].
const double kOutputGasDensityOnStandartConditions = 0.6865365;
/// Содержание углекислого газа, [б.р.].
const double kOutputGasCarbonDioxidePart = 0.0;
/// Содержание азота, [б.р.].
const double kOutputGasNitrogenPart = 0.0;
/// Давление, [МПа].
const double kOutputGasPressure = 3.0;

/** Результат Весты для описанной выше конфигурации - q = 387.84 [м3/с].
Результат моего расчёта - 385.8383, что похоже на Весту, но не совпадает.
Тем не менее считаем это решение за эталон - при дальнейших изменениях, тесты
будут сигнализировать, что что-то изменилось, будем смотреть и разбираться.*/
const double kTestPipeQuantity = 385.8427; /// Расчитаннй мной расход.
const double kTestPipeQuantityPrecision = 0.0001; /// Точность расчёта.

// Forward-declarations.
struct  Gas;
struct  PassportPipe;

class   WorkParams;
class   CalculatedParams;

template<class VectorElement>
void LoadSerializableVectorFromFile( 
    std::string file_name, 
    std::vector<VectorElement> *vec) {
  std::ifstream ifs(file_name);
  assert(ifs.good());
  boost::archive::xml_iarchive ia(ifs);
  ia >> BOOST_SERIALIZATION_NVP(*vec);
}


void FillTestPassportPipe(PassportPipe* passport);
PassportPipe MakeTestPassportPipe();

void FillTestGasIn(Gas* gas);
Gas MakeTestGasIn();

void FillTestGasOut(Gas* gas);
Gas MakeTestGasOut();

std::vector<std::string> FileAsVectorOfStrings(std::string filename);

const std::string path_to_vesta_files = "C:\\Enisey\\data\\saratov_gorkiy\\";
static const std::string etalon_saratov_gorkiy_balance = 
  "C:\\Enisey\\out\\SaratovGorkiy\\etalon_balance_find.txt";

void CompareGTSDisbalancesFactToEtalon(
    const std::vector<double> &abs_disbalances,
    const std::vector<int> &int_disbalances);