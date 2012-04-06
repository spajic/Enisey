/** \file gas_transfer_system.h
Класс GasTransferSystem - придставление газотранспортной системы (ГТС).*/
#pragma once
#include <string>
// Forward-declarations.
class GraphBoost;

/** Класс представляет объект ГТС.*/
class GasTransferSystem {
 public:
  /** Создаёт объект GraphBoost g_.*/
  GasTransferSystem();
  /** Удаляет объект GraphBoost g_.*/
  ~GasTransferSystem();
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
  /**Произвести смешивание газовых потоков в вершинах.
  Выполняется в топологическом порядке, влияет на состав газа, T вершин.
  Не влияет на P, Q вершин.*/
  void MixVertices();
 private:
  GraphBoost *g_; ///< Внутреннее представление графа - GraphBoost.
  /** Первоначальное смешение газовых потоков. Просто "протягиваем"
  состав газа от входа к выходам.*/
  void MakeInitialMix();
};
