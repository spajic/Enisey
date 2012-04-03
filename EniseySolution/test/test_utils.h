/** \file test_utils.h
����������� ������� ��� ������������. */
#pragma once

/// ��������� ��� �������� �����.
/// ���������� ������� �����, [��].
const float kInnerPipeDiameter = 1000.0; 
/// �������� ������� �����, [��].
const float kOuterPipeDiameter = 1020.0;
/// ����� �����, [��].
const float kPipeLength = 100.0;
/// ����������� ����������� ����� � ������� ������, [].
const float kHeatExchangeCoefficient = 1.3;
/// ����������� �������������� �������������, [].
const float kHydraulicEfficiencyCoefficient = 0.95;
/// ����������� �� ������������ �������� [���].
const float kMaximumPressure = 100.0;
/// ����������� �� ����������� �������� [���].
const float kMinimumPressure = 1.0;
/// ����������� ������������� ������������� [].
const float kRoughnessCoefficient = 0.03;
/// ����������� ���������� ����� [�].
const float kEnvironmentTemperature = 280.15;

/// ��������� ��� ��������� �������� ������.
/// ��������� ��� ����������� ��������, [��/�3].
const float kInputGasDensityOnStandartConditions = 0.6865365;
/// ���������� ����������� ����, [�.�.].
const float kInputGasCarbonDioxidePart = 0.0;
/// ���������� �����, [�.�.].
const float kInputGasNitrogenPart = 0.0;
/// ��������, [���].
const float kInputGasPressure = 5.0;
/// �����������, [�].
const float kInputGasTemperature = 293.15;
/// ������, [�3/���].
const float kInputGasQuantity = 387.843655734;

/// ��������� ��� ���������� �������� ������.
/// ��������� ��� ����������� ��������, [��/�3].
const float kOutputGasDensityOnStandartConditions = 0.6865365;
/// ���������� ����������� ����, [�.�.].
const float kOutputGasCarbonDioxidePart = 0.0;
/// ���������� �����, [�.�.].
const float kOutputGasNitrogenPart = 0.0;
/// ��������, [���].
const float kOutputGasPressure = 3.0;

/** ��������� ����� ��� ��������� ���� ������������ - q = 387.84 [�3/�].
��������� ����� ������� - 385.8383, ��� ������ �� �����, �� �� ���������.
��� �� ����� ������� ��� ������� �� ������ - ��� ���������� ����������, �����
����� ���������������, ��� ���-�� ����������, ����� �������� � �����������.*/
const float kTestPipeQuantity = 385.8383; /// ���������� ���� ������.
const float kTestPipeQuantityPrecision = 0.0001; /// �������� �������.

// Forward-declarations.
struct PassportPipe;
struct Gas;

void FillTestPassportPipe(PassportPipe* passport);
PassportPipe MakeTestPassportPipe();

void FillTestGasIn(Gas* gas);
Gas MakeTestGasIn();

void FillTestGasOut(Gas* gas);
Gas MakeTestGasOut();