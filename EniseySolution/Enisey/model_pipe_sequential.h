/** \file model_pipe_sequential.h
Класс ModelPipeSequential - модель последовательного счёта для трубы.*/
#pragma once
#include "gas.h"
#include "passport_pipe.h"

/** Модель последовательного счёта для трубы. 
Используется в ManagerEdgeModelPipeSequential в качестве "рабочей лошадки". */
class ModelPipeSequential {
 public:
  /// Конструктор по умолчанию. Ничего не делает.
  ModelPipeSequential();
  /// Приводит переданный объект passport к типу PassportPipe и сохраняет.
  /// \todo Лучше пусть здесь будет сразу PassportPipe.
  ModelPipeSequential(const Passport* passport);
  /// Установить объект Газ на входе трубы.
  void set_gas_in(const Gas* gas);
  /// Установить объект Газ на выходе трубы.
  void set_gas_out(const Gas* gas);
  /// Произвести расчёт расхода q и tвых по (p_вх, t_вх, состав_газа_вх, p_вых)
  void Count();
  /// Поулчить расчитанный расход.
  float q();
 private:
  void DetermineDirectionOfFlow(); ///< Определить направление потока.
  bool direction_is_forward_; ///< Направление потока - прямое или обратное.
  float q_; ///< Расход.
  PassportPipe passport_; ///< Паспорт трубы.
  Gas gas_in_; ///< Объект Газ на входе.
  Gas gas_out_; ///< Объект Газ на выходе.
};