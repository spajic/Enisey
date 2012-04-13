/** \file gas_transfer_system_i.h
Класс GasTransferSystemI - абстракнтый интерфей ГТС.*/
#pragma once
#include <string>
#include <vector>
#include <map>
// Forward-declarations.
class GraphBoost;
class SlaeSolverI;
class ManagerEdge;

/** Класс представляет абстрактный интерфейс ГТС.*/
class GasTransferSystemI {
 public:
  /** Создаёт объект GraphBoost g_.*/
  //GasTransferSystemI() = 0;
  /** Удаляет объект GraphBoost g_.*/
  //virtual ~GasTransferSystemI() = 0;
  /** Сопоставить узлам графа номера уравнений СЛАУ.
  Узлам, для которых PIsReady() == false номера от и выше.
  Узлам PIsReady не участвуют в расчёте, для них номер = -1.*/
  virtual void SetSlaeRowNumsForVertices() = 0;
  /** Формирование СЛАУ. */
  virtual void FormSlae() = 0;
  /** Выполнить решение составленной СЛАУ. 
  В результате заполняется вектор DeltaP_.*/
  virtual void SolveSlae() = 0;
  /** Получить новую итерацию.*
  Выполнить смешение всех вершин.
  Имея вектор DeltaP_ всем вершинам задать приращение.
  Произвести перерасчёт всех рёбер.
  Дисбаланс должен уменьшиться.*/
  virtual void CountNewIteration(double g) = 0;
  /** Суммарный дисбаланс в системе. Сумма дисбалансов всех вершин.*/
  virtual double CountDisbalance() = 0;
  virtual int GetIntDisbalance() = 0;
  /** Загрузить граф из файлов Весты, находящихся в папке path.
  Путь передаётся с последним слешем, например "C:\\vesta\\files\\"*/
  virtual void LoadFromVestaFiles(std::string const path) = 0;
  /** Выести граф в файл filename. */
  virtual void const WriteToGraphviz(std::string const filename) = 0;
  /** Выполнить расчёт начального приближения ограничений, давлений 
  и температур.*/
  virtual void MakeInitialApprox() = 0;
  /** Произвести расчёт всех рёбер графа. Т.е. во всех рёбрах должны быть
  доступны расход и производные расхода.*/
  virtual void CountAllEdges() = 0;
  /** Произвести смешивание газовых потоков в вершинах.
  Выполняется в топологическом порядке, влияет на состав газа, T вершин.
  Не влияет на P, Q вершин.*/
  virtual void MixVertices() = 0;
  /** Установить SlaeSolver.*/
  virtual void set_slae_solver(SlaeSolverI *slae_slover) = 0;
  /** Установить ManagetEdge.*/
  virtual void set_manager_edge(ManagerEdge* manager_edge) = 0;
};
