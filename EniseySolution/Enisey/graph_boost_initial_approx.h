/** \file graph_boost_initial_aprox.h
Функции для задания начальных приближений в графе ГТС GraphBoost.*/
#pragma once
// Forward-declarations.
class GraphBoost;

void FindOverallMinAndMaxPressureConstraints(
  GraphBoost *graph, 
  float *overall_p_max,
  float *overall_p_min);

/**Задать для графа поле давлений - начальное приближение.
Начальное приближение должно удовлетворять следующим условиям:<pre>
 1. Давления должны убывать от начала к концу графа.
 2. Давления должны соблюдать ограничения на макс. и мин. давления дуги. </pre>
\param graph Граф, для которого нужно задать начальное приближение.*/
void SetInitialApproxPressures(GraphBoost *graph);

/**Задать для всех вершин графа ограничения по давлению.<pre>
Входные данные для алгоритма:
 1. У каждой трубы есть в пасспорте p_min, p_max - паспортные ограничения.
Алгоритм расчёта: 
 1. В топологическом порядку "сверху вниз" для каждго узла его ограничение на
    максимальное давление 
    p_max = min( min( passport.p_max исх-х труб ), min( p_max вх-х узлов ) ).
    Представьте: из вершины исходит два ребра с p_max1 = 20 и p_max2 = 30.
    Понятно, что максимальное давление в узле не может быть больше 20.
    Так же если у узла есть входящие узлы, для которых ограничение p_max уже
    расчитано, ограничение текущего узла дложно быть меньше всех этих огр-й.
 2. В обратном топологическом порядке "снизу вверх" для каждого узла его
    ограничение на мин давление 
    p_min = max( max( passport.p_min вх-х рёбер), ( max( p_min ) исх-х узлов ).
    Представьте: в узел входит два ребра с p_min1 = 5, p_min2 = 10.
    Ясно, что p_min не может быть меньше 10.
    Так же если у узла есть исходящие узлы, для которых ограничение p_min уже
    расчитано, ограничение текущего узла дложно быть больше всех этих огр-й.
 3. Для входов/выходов InOutGRS.dat содержит только p_min. Поэтому 
    a) для выходов задаём p_min расчитаное на шаге 2, либо из InOutGRS, 
    если оно задано.
    б) для входов задаём p_min из InOutGRS, если задано, либо минимальное
    ограничение среди всех объектов ГТС.
 4. Для выходов ограничение p_max задаём максимальное ограничение среди всех
    объектов ГТС.
!!. Пока задаём для входов и выходов p_min, p_max соответственно минимальное
    и максимальное ограничение среди всех объектов ГТС.
</pre>
*/
void SetPressureConstraintsForVertices(
    GraphBoost *graph,
    float overall_p_min,
    float overall_p_max );

/** Проверить непротиворечивость задания ограничений на давление в графе.<pre>
 1. overall_p_min, overall_p_max должны быть > 0.
 2. Все p_min, p_max должны быть в интервале [overall_p_min, overall_p_max]. 
 3. Для каждого узла интервал [p_min, p_max] должен быть не пустым.</pre>*
Возвращает true, если всё корректно. */
bool ChechPressureConstraintsForVertices(
    GraphBoost *graph,
    float overall_p_min,
    float overall_p_max );