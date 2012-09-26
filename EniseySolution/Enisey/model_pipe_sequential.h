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
  /// Получить объект Газ на входе трубы.
  const Gas& gas_in();
  /// Установить объект Газ на выходе трубы.
  void set_gas_out(const Gas* gas);
  /// Получить объект Газ на выходе трубы.
  const Gas& gas_out();
  /** Произвести расчёт расхода q и tвых по (p_вх, t_вх, состав_газа_вх, p_вых)
  То есть для того, чтобы можно было выполнить Count нужно предварительно 
  выполнить set_gas_in, set_gas_out, откуда будут взяты необходимые параметры
  на входе и выходе. 
  Если p_вх < p_вых, то труба считается реверсивной, газ течёт в обратную
  сторону. В этом случае gas_out считается входом, а gas_in - выходом, расход
  возвращается отрицательный.
  Так же эта функция выполняет расчёт производных: dq_dp_in, dq_dp_out.
  dq_dp = ( q(p + eps) - q(p) ) / eps.
  */
  void Count();
  double q(); ///< Поулчить расчитанный расход.
  double dq_dp_in(); ///< Получить производную по Pвх.
  double dq_dp_out(); ///< Получить производную по Pвых.
  bool IsReverse(); ///< Направление потока обратное? (м.б. прямым и обратным)
 private:
  double FindQWithDeltaPIn(double delta_p_in, int segments);
  double FindQWithDeltaPOut(double delta_p_out, int segments);
  /// Вызов функции расчёта q с стуктурированными параметрами.
  void CallFindSequentialQ(
      const Gas &gas_in,
      const Gas &gas_out,
      const PassportPipe &passport,
      const int number_of_segments,
      double *t_out,
      double *q_out); 
  double q_; ///< Расход.
  double dq_dp_in_; ///< Производная q по Pвх.
  double dq_dp_out_; ///< Производная q по Pвых.
  PassportPipe passport_; ///< Паспорт трубы.
  Gas gas_in_; ///< Объект Газ на входе.
  Gas gas_out_; ///< Объект Газ на выходе.
};