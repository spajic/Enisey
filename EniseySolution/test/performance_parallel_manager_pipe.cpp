/** \file performance_parallel_manager_pipe.cpp
Замер производительности классов ParallelManagerPipe.
Нужно замерить все комбинации следующих аспектов:
Реализация менеджера: SingleCore, OpenMP, CUDA, ICE;
Имплементация сервера ICE: SingleCore, OpenMP, CUDA;
Расположение Сервера ICE: локально, в AWS;
Тестовый пример: Саратов, 10x, 100x, 1000x Саратов;
Использование: однократный расчёт, итеративная смена рабочих параметров;
Количество итераций при итеративном расчёте: 10, 100.
При работе тест должен логгировать тайминги основных событий.
ToDo:
0. Проверить корректность работы ParallelManagerPipeI при итеративном вызове
   SetWorkParams.
1. Написать тест для объекта типа ParallelManagerSingleCore.
    - Расчёт 100x Саратова [10 раз и усреднить] 
    - Расчёт 100х Саратова по 10 итераций [10 раз и усреднить]
    [100, или 1000000 - подобрать так, чтобы делалось заметное время, но 
    довольно быстро - чтобы было куда ускоряться] 
    [Параметры должны задаваться через конфигурационный файл]
2. Сделать тест обобщённым.
3. Использовать тест в рамках Запускающего теста с различными реализацями
   ParallelManagerPipeI - OpenMP, CUDA, ICE с различными локальными реализациями.
4. Запустить сервер ICE в облаке AWS, настроить доступ к нему и затестить с ним -
   тоже с различными реализациями.
*/

