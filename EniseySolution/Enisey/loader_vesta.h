/** \file loader_vesta.h
Функции для загрузки файлов Весты.
Функции, представленные в данном файле преобразуют файлы Весты в более удобное
внутреннее представление. */
#pragma once
#include <string>
#include <hash_map>
#include "passport_pipe.h"

/** Содержит информацию о рёбрах (объектах), извлекаемую из файлов
Весты.*/
struct EdgeData {
  EdgeData(); ///< Устанавливает для всех членов значение -1.
  int id_vesta; ///< id ребра из файла Весты MatrixConections.dat
  int in_vertex_id; ///< id узла, с которого начинается ребро.
  int out_vertex_id; ///< id узла, которым заканчивается ребро.
  PassportPipe passport; ///< Паспорт трубы.
  int edge_type; ///< Тип ребра (расшифровка у ф-ии LoadMatrixConnections).
  int pipe_type; ///< Тип трубы (расшифровка у ф-ии LoadPipeLines).
};
/** Содержит инф-ю о входах/выходах или отборах ГТС из файлов Весты.
*/
struct InOutData {
  InOutData(); ///< Устанавливает для всех членов значение -1.
  int id; ///< id чего?
  float q; ///< Расход входа/выхода.
  float temp; ///< Температура газа.
  float den_sc; ///< Плотность газа при стд. усл.
  float pressure; ///< Давление газа.
  float n2; ///< Содержание азота N2.
  float co2; ///< Содержание двуокиси угдерода CO2.
  float heat_making; ///< Низшая теплотворная способность.
  float id_in_out; ///< id входа/выхода или отбора.
  float min_p; ///< Мин. допустимое давление газа на ГРС.
};
/** Содержит инф-ю об узлах, извлекаемую из файлов Весты. */
struct VertexData {
  VertexData(); ///< Устанавливает для всех членов значение -1.
  int id_vesta;	///< id узла из файлов Весты
  int id_graph; ///< id, который выдал граф при добавлении в него вершины.
  std::list<int> edges_id_out; ///< Список id Весты исходящих из узла рёбер.
  std::list<int> edges_id_in; ///< Список id Весты входящих в узел рёбер.
  std::list<InOutData> in_outs; ///< Список входов/выходов/отборов данного узла
};
/** Содержит обобщённую инф-ю, собираемую из файлов Весты.
Здесь участвуют файлы MatrixConnections, PipeLine, InOutGRS.*/
struct VestaFilesData {
  VestaFilesData(); ///< Устанавливает для всех членов значение -1.
  int num_edges; ///< Общее количество объектов (рёбер) в ГТС.
  int num_nodes; ///< Общее количество узлов в ГТС.
  int elems_per_string; ///< Количество элементов в строке MatrixConnections.
  stdext::hash_map<int, EdgeData> edges_hash; ///< хеш (id_ребра, Ребро).
  stdext::hash_map<int, VertexData> vertices_hash; ///< хеш (id_узла, Узел).
};

/** Считывает инф-ю из MatrixConnections.dat в структуру VestaFilesData.
Файл MatrixConnections.dat содержит инф-ю по расчётной схеме ГТС.
Файл имеет следующий формат:<pre>
1-я строка: кол-во_объектов(рёбер) кол-во_узлов кол-во_эл-в_в_строке.
Далее в цикле по объектам (рёбрам) расчётной схемы ГТС строки из двух 
чисел: 
  идентификатор объекта (ребра); 
  тип ребра(1 - труба, 2 - КЦ, 3 - кран регулятор, или байпас, 4 - ОЛП/ПУ
            5 - АВО, 6 - ГПА, 7 - кран МСП, 8 - КЦ (потоковая модель)).
  </pre>
Далее в цикле по узлам расчётной схемы ГТС: строки из кол-во_эл-в_в_строке <br>
  Ид_узла ид_вх/вых_объекта(если ребро входит в узел - знак "-") нули.*/
void LoadMatrixConnections(std::string file_name, VestaFilesData *vsd);

/** Считывает инф-ю из PipeLine.dat в структуру VestaFilesData.
Файл PipeLine.dat содержит информацию по трубам. <pre>
 1-я строка: общее кол-во труб (м.б. больше, чем участвует в расчёте).
 Далее в цикле по трубам:
  Идентификатор объекта (ребра-трубы)
   Тип_трубы (0-магистраль, 2-отвод, 3-дюкер/лупинг)
    номер нитки МГ
     гидравлическая эф-ть
      к-т теплопередачи от газа в окр. среду [вт/(м2*град)]
       температора окр. среды [C]
        мин. допустимое давл-е газа в трубе [ата]
         макс. доп. давл-е газа в трубе [ата]
          внутренний диаметр трубы [мм]
           наружный димаетр трубы [мм]
            длина трубы [км]
             к-т шерохов-ть трубы [мм]
              разность высот [м]
 Пример строки: 686 3 43337 0.7007 1.3 2.5 20 55 515 530 4.000 0.03 0</pre>*/
void LoadPipeLines(std::string file_name, VestaFilesData *vsd);

/** Считывает инф-ю из InOutGRS.dat в структуру VestaFilesData.
<pre>
Файл InOutGRS.dat содержит информацию по входам/выходам ГТС и отборам.
1-я строка - общее кол-во входов/выходов ГТС, попутных отборов газа.
Далее в цикле по входам/выходам в ГТС
  Идентификатор узла
   Расход газа, если отбор - то со знаком "-", если приток - "+" [млн м3/сут]
    Температура газа [C]
     Плотность газа при стд. усл. [кг/м3]
      Давление газа [ата]
       Содержание азота N2 [%]
        Содержание двуокиси углерода CO2 [%]
         Низшая теплотворная способность газа [кДж/м3]
          Признак контр. давления (0-обычный приток, 1-фиктивный отбор
              служит для задания опорного давления газа в узле,
              2 - отвод отключен, но в нестац. режиме может быть включён,
              3 - фиктивный отвод служит для резки ТС и задания P и T.
           Идентификатор входа/выхода или ГРС
            Мин. допустимое давление газа на ГРС
             Идентификатор нитки
              Содержание конденсата (влаги) [г/м3]
               Pст [ата]
                A
                 B
                  Qмак [млн м3/сут]
Примечание: Фиктивный отбор служит для задания опорного давления газа в
    узле. Если номер узла фиктивного отбора совпадает с узлом для
    фактического отбора/притока, для которого задаётся расход газа, то
    формируются две строки для фактического и фиктивного отбора.
Примечание:
   Для притоков - задаются Q>0 или P, T, den_sc, N2, Co2
   Для отводов - Q или P
   Если отбор с магистрали (без трубопровода отвода) - только Q
Примечание: явно есть "лишние" по сравнению с описанием числа.
Примеры:
  296 0.00000000 18.00 0.68991 34.00 0 0 34541 0 578 0 0 0 0.0 0.0 0.0 0.0
  238 -1.58500000 0.000 0.0 0.000 0.0 0.0 0.0 0 583 0 0 0.0 0.0 0.0 0.0 0.0
</pre>
 \todo Уточнить у @МСК значение новых параметров Pст, A, B, Qмак.*/
void LoadInOutGRS(std::string file_name, VestaFilesData *vsd);