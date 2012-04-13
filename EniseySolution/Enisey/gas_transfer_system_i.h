/** \file gas_transfer_system_i.h
����� GasTransferSystemI - ����������� �������� ���.*/
#pragma once
#include <string>
#include <vector>
#include <map>
// Forward-declarations.
class GraphBoost;
class SlaeSolverI;
class ManagerEdge;

/** ����� ������������ ����������� ��������� ���.*/
class GasTransferSystemI {
 public:
   /// ��������� ������������ �������.
   virtual void PeroformBalancing(
     const std::vector<std::string> &MatrixConnectionsFile,
     const std::vector<std::string> &InOutGRSFile,
     const std::vector<std::string> &PipeLinesFile,
     std::vector<std::string> *ResultFile,
     std::vector<double> *AbsDisbalances,
     std::vector<int> *IntDisbalances
  ) = 0;
  /** ������ ������ GraphBoost g_.*/
  //GasTransferSystemI() = 0;
  /** ������� ������ GraphBoost g_.*/
  //virtual ~GasTransferSystemI() = 0;
  /** ����������� ����� ����� ������ ��������� ����.
  �����, ��� ������� PIsReady() == false ������ �� � ����.
  ����� PIsReady �� ��������� � �������, ��� ��� ����� = -1.*/
  virtual void SetSlaeRowNumsForVertices() = 0;
  /** ������������ ����. */
  virtual void FormSlae() = 0;
  /** ��������� ������� ������������ ����. 
  � ���������� ����������� ������ DeltaP_.*/
  virtual void SolveSlae() = 0;
  /** �������� ����� ��������.*
  ��������� �������� ���� ������.
  ���� ������ DeltaP_ ���� �������� ������ ����������.
  ���������� ���������� ���� ����.
  ��������� ������ �����������.*/
  virtual void CountNewIteration(double g) = 0;
  /** ��������� ��������� � �������. ����� ����������� ���� ������.*/
  virtual double CountDisbalance() = 0;
  virtual int GetIntDisbalance() = 0;
  /** ��������� ���� �� ������ �����, ����������� � ����� path.
  ���� ��������� � ��������� ������, �������� "C:\\vesta\\files\\"*/
  virtual void LoadFromVestaFiles(std::string const path) = 0;
  /** ������ ���� � ���� filename. */
  virtual void const WriteToGraphviz(std::string const filename) = 0;
  /** ��������� ������ ���������� ����������� �����������, �������� 
  � ����������.*/
  virtual void MakeInitialApprox() = 0;
  /** ���������� ������ ���� ���� �����. �.�. �� ���� ����� ������ ����
  �������� ������ � ����������� �������.*/
  virtual void CountAllEdges() = 0;
  /** ���������� ���������� ������� ������� � ��������.
  ����������� � �������������� �������, ������ �� ������ ����, T ������.
  �� ������ �� P, Q ������.*/
  virtual void MixVertices() = 0;
  /** ���������� SlaeSolver.*/
  virtual void set_slae_solver(SlaeSolverI *slae_slover) = 0;
  /** ���������� ManagetEdge.*/
  virtual void set_manager_edge(ManagerEdge* manager_edge) = 0;
};
