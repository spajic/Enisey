/** \file test_utils.h
����������� ������� ��� ������������. */
#pragma once
#include <vector>

/// ��������� ��� �������� �����.
/// ���������� ������� �����, [��].
const double kInnerPipeDiameter = 1000.0; 
/// �������� ������� �����, [��].
const double kOuterPipeDiameter = 1020.0;
/// ����� �����, [��].
const double kPipeLength = 100.0;
/// ����������� ����������� ����� � ������� ������, [].
const double kHeatExchangeCoefficient = 1.3;
/// ����������� �������������� �������������, [].
const double kHydraulicEfficiencyCoefficient = 0.95;
/// ����������� �� ������������ �������� [���].
const double kMaximumPressure = 100.0;
/// ����������� �� ����������� �������� [���].
const double kMinimumPressure = 1.0;
/// ����������� ������������� ������������� [].
const double kRoughnessCoefficient = 0.03;
/// ����������� ���������� ����� [�].
const double kEnvironmentTemperature = 280.15;

/// ��������� ��� ��������� �������� ������.
/// ��������� ��� ����������� ��������, [��/�3].
const double kInputGasDensityOnStandartConditions = 0.6865365;
/// ���������� ����������� ����, [�.�.].
const double kInputGasCarbonDioxidePart = 0.0;
/// ���������� �����, [�.�.].
const double kInputGasNitrogenPart = 0.0;
/// ��������, [���].
const double kInputGasPressure = 5.0;
/// �����������, [�].
const double kInputGasTemperature = 293.15;
/// ������, [�3/���].
const double kInputGasQuantity = 387.843655734;

/// ��������� ��� ���������� �������� ������.
/// ��������� ��� ����������� ��������, [��/�3].
const double kOutputGasDensityOnStandartConditions = 0.6865365;
/// ���������� ����������� ����, [�.�.].
const double kOutputGasCarbonDioxidePart = 0.0;
/// ���������� �����, [�.�.].
const double kOutputGasNitrogenPart = 0.0;
/// ��������, [���].
const double kOutputGasPressure = 3.0;

/** ��������� ����� ��� ��������� ���� ������������ - q = 387.84 [�3/�].
��������� ����� ������� - 385.8383, ��� ������ �� �����, �� �� ���������.
��� �� ����� ������� ��� ������� �� ������ - ��� ���������� ����������, �����
����� ���������������, ��� ���-�� ����������, ����� �������� � �����������.*/
const double kTestPipeQuantity = 385.8427; /// ���������� ���� ������.
const double kTestPipeQuantityPrecision = 0.0001; /// �������� �������.

// Forward-declarations.
struct PassportPipe;
struct Gas;

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