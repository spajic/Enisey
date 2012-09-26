/** \file graph_boost_load_from_vesta.h
Функция для заполнения графа GraphBoost по структуре VestaFilesData.*/
#pragma once
// Forward-declarations.
class GraphBoost;
struct VestaFilesData;
/** Заполняет передаваемый граф вершинами и рёбрами на основании передаваемой
структуры VestaFilesData. 
\param graph Указатель на граф, который нужно заполнить.
\param vfd Указатель на структуру VestaFilesData, содержащую информацию,
считанную из файлов Весты.*/
void GraphBoostLoadFromVesta(GraphBoost *graph, VestaFilesData *vfd);
/** Записать по указателю soted_graph граф эквивалентный тому, что передан в g,
только у которого вершины и рёбры отсортированы в топологическом порядке.
Так, что родные итераторы совпадают с топологическими.*/
void SortGraph(GraphBoost &g, GraphBoost *sorted_graph);