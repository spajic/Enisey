/** \file gas_transfer_system.h
Класс GasTransferSystem - придставление газотранспортной системы (ГТС).*/
#pragma once
#include <string>
#include <vector>
#include <map>
// Forward-declarations.
class GraphBoost;
class SlaeSolverI;

/** Класс представляет объект ГТС.*/
class GasTransferSystem {
 public:
  /** Создаёт объект GraphBoost g_.*/
  GasTransferSystem();
  /** Удаляет объект GraphBoost g_.*/
  ~GasTransferSystem();
  /** Сопоставить узлам графа номера уравнений СЛАУ.
  Узлам, для которых PIsReady() == false номера от и выше.
  Узлам PIsReady не участвуют в расчёте, для них номер = -1.*/
  void SetSlaeRowNumsForVertices();
  /** Формирование СЛАУ. */
  void FormSlae();
  /** Выполнить решение составленной СЛАУ. 
  В результате заполняется вектор DeltaP_.*/
  void SolveSlae();
  /** Получить новую итерацию.*
  Выполнить смешение всех вершин.
  Имея вектор DeltaP_ всем вершинам задать приращение.
  Произвести перерасчёт всех рёбер.
  Дисбаланс должен уменьшиться.*/
  void CountNewIteration(double g);
  /** Суммарный дисбаланс в системе. Сумма дисбалансов всех вершин.*/
  double CountDisbalance();
  int GetIntDisbalance();
  /** Загрузить граф из файлов Весты, находящихся в папке path.
  Путь передаётся с последним слешем, например "C:\\vesta\\files\\"*/
  void LoadFromVestaFiles(std::string const path);
  /** Выести граф в файл filename. */
  void const WriteToGraphviz(std::string const filename);
  /** Выполнить расчёт начального приближения ограничений, давлений 
  и температур.*/
  void MakeInitialApprox();
  /** Произвести расчёт всех рёбер графа. Т.е. во всех рёбрах должны быть
  доступны расход и производные расхода.*/
  void CountAllEdges();
  /** Произвести смешивание газовых потоков в вершинах.
  Выполняется в топологическом порядке, влияет на состав газа, T вершин.
  Не влияет на P, Q вершин.*/
  void MixVertices();
  /** Установить SlaeSolver.*/
  void set_slae_solver(SlaeSolverI *slae_slover);
 private:
   /// Количество строк в СЛАУ.
  int slae_size_;
  GraphBoost *g_; ///< Внутреннее представление графа - GraphBoost.
  std::map<std::pair<int, int>, double> A_; ///< Матрица СЛАУ.
  std::vector<double> B_; ///< Вектор правых частей СЛАУ.
  std::vector<double> DeltaP_; ///< Вектор решений СЛАУ.
  /** Первоначальное смешение газовых потоков. Просто "протягиваем"
  состав газа от входа к выходам.*/
  void MakeInitialMix();
  SlaeSolverI *slae_solver_;
};
