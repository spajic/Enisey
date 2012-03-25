// EXPECT(expected,actual) - не фатальная ошибка, предпочтительно использовать.
// ASSERT - фатальная ошибка, использовать, когда продолжение не имеет смысла.
//TEST(test_case_name, test_name) {
//  EXPECT_EQ(1, 2) << "Nonsense!";
//  ASSERT_EQ(2, 3) << "I can't take this!";
//  EXPECT_EQ(3, 4) << "This is impossible!";
//}
// The fixture for testing class Foo.
//class FooTest : public ::testing::Test {
//protected:
//  FooTest() {} // You can do set-up work for each test here.
//  virtual ~FooTest() { } // You can do clean-up work that doesn't throw exceptions here.
//  // If the constructor and destructor are not enough for setting up
//  // and cleaning up each test, you can define the following methods:
//  virtual void SetUp() { }
//    // Code here will be called immediately after the constructor (right
//    // before each test).
//  virtual void TearDown() { }
//    // Code here will be called immediately after each test (right
//    // before the destructor).

//  // Objects declared here can be used by all tests in the test case for Foo.
//};
//// Tests that the Foo::Bar() method does Abc.
//TEST_F(FooTest, MethodBarDoesAbc) {
//  const string input_filepath = "this/package/testdata/myinputfile.dat";
//  const string output_filepath = "this/package/testdata/myoutputfile.dat";
//  Foo f;
//  EXPECT_EQ(0, f.Bar(input_filepath, output_filepath));
//}

#include "gtest/gtest.h"
#include "test_utils.h"

#include "gas.h"
#include "functions_pipe.h"
#include "passport_pipe.h"
TEST(Gas, GasCountFunctions)
{
  // Тестируем правильность работы газовых функций.
  // Тест такой - в Весте выставяляем использование НТП-2006
  // В локальной задаче смотрим вычисленные Вестой значения.
  // Если получается довольно похоже, принимаем рассчитанные мной значения
  // за эталон - и сохраняем для дальнейшего сравнения с ними.
  // В Весте показываются значения для z, c, mju.

  // Задаём свойства газа, для которого проводим тестирование.
  GasCompositionReduced composition;
  composition.density_std_cond = 0.6865365; // [кг/м3]
  composition.co2 = 0;
  composition.n2 = 0;
  GasWorkParameters params;
  params.p = 2.9769; // [МПа]
  params.t = 279.78; // [К]
  params.q = 387.843655734; // [м3/сек]

  // Вычисление базовых характеристик газа только по составу
  // t_pseudo_critical = 194.07213 [К]
  // p_pseudo_critical = 4.6355228 [МПа]
  // z_standart_conditions = 0.99798650 [б.р.]
  // r_standart_conditions = 503.37848 [Дж / (кг * К)]
  float t_pseudo_critical = FindTPseudoCritical(
    composition.density_std_cond, composition.co2, composition.n2);
  float p_pseudo_critical = FindPPseudoCritical(
    composition.density_std_cond, composition.co2, composition.n2);
  float z_standart_conditions = FindZStandartConditions(
    composition.density_std_cond, composition.co2, composition.n2);
  float r_standart_conditions = FindRStandartConditions(
    composition.density_std_cond);

  // Вычисление характеристик газа при рабочем давлении и температуре
  // p_reduced = 0.64219296
  // t_reduced = 1.4416289
  // c = 2372.5037		(значение в Весте Cp = 2310.1)
  // di = 5.0115082e-6
  // mju = 1.0930206e-5	(значение в Весте mju = 1.093e-5)
  // z = 0.91878355		(значение в Весте z = 0.91878)
  // ro = 23.002302
  float p_reduced = FindPReduced(params.p, p_pseudo_critical);
  float t_reduced = FindTReduced(params.t, t_pseudo_critical);
  float c = FindC(t_reduced, p_reduced, r_standart_conditions);
  float di = FindDi(p_reduced, t_reduced);
  float mju = FindMju(p_reduced, t_reduced);
  float z = FindZ(p_reduced, t_reduced);
  float ro = FindRo(composition.density_std_cond, params.p, params.t, z);

  // Проверим на адекватность так же функции, которые зависят и от пасспорта трубы
  PassportPipe passport;
  FillTestPassportPipe(&passport);

  // re = 31017164.0
  // lambda = 0.010797811
  float re = FindRe(params.q, composition.density_std_cond, mju, passport.d_inner_);
  float lambda = FindLambda(re, passport.d_inner_, passport.roughness_coeff_, passport.hydraulic_efficiency_coeff_);

  // Видно, что значения mju и z практически совпали с Вестой,
  // с немного отличается. 
  // Принимаем вычисленные решения за эталон и делаем EXPECT'ы
  // Если значения вдруг станут вычислятся по другому, тест даст нам знать.

  // Задаём точность сравнения eps
  float eps = 1.0e-4;

  // Проверки базовых параметров
  EXPECT_LE(abs(t_pseudo_critical - 194.07213), eps);
  EXPECT_LE(abs(p_pseudo_critical - 4.6355228), eps);
  EXPECT_LE(abs(z_standart_conditions - 0.99798650), eps);
  EXPECT_LE(abs(r_standart_conditions - 503.37848), eps);

  // Проверки параметров при рабочих условиях
  EXPECT_LE(abs(p_reduced - 0.64219296), eps);
  EXPECT_LE(abs(t_reduced - 1.4416289), eps);
  EXPECT_LE(abs(c - 2372.5037), eps);
  EXPECT_LE(abs(di - 5.0115082e-6), eps);
  EXPECT_LE(abs(mju - 1.0930206e-5), eps);
  EXPECT_LE(abs(z - 0.91878355), eps);
  EXPECT_LE(abs(ro - 23.002302), eps);

  // Проверки для re и lambda
  // Для re задаём свою точность, потому что число большое
  float eps_re = 1;
  EXPECT_LE(abs(re - 31017164.0), eps_re);
  EXPECT_LE(abs(lambda - 0.010797811), eps);
}

TEST(PipeSequential, CountSequentialOut)
{
  // Тест расчёта трубы методом последовательного счёта
  // Тестировать будем так - зададим входные данные, получим расчётные рез-ты
  // и сравним их с рез-ми Весты для таких же входных данных.
  // Если результаты окажутся похожими, сохраним их и в дальнейшем будем
  // использовать в качестве эталона.

  // Задаём пасспортные свойства трубы.
  PassportPipe passport;
  FillTestPassportPipe(&passport);

  // Задаём свойства газа на входе.
  GasCompositionReduced composition;
  GasWorkParameters params_in;
  composition.density_std_cond = 0.6865365; // [кг/м3]
  composition.co2 = 0;
  composition.n2 = 0;
  params_in.p = 5; // [МПа]
  params_in.t = 293.15; // [К]
  params_in.q = 387.843655734; // [м3/сек]

  // Производим расчёт параметров газа на выходе.
  float p_out;
  float t_out;

  float number_of_segments = 10;
  float length_of_segment = passport.length_ / number_of_segments;
  // Получены результаты p_out = 2.9721224, t_out = 280.14999 (В Весте - p_out = 2.9769, t_out = 279.78)
  // Результаты очень похожи на Весту, принимаем их за эталон.
  // Видна особенность - у меня температура выходит на Тос, а в Весте - продолжает падать.
  // ToDo: разобраться точно с этой особенностью.
  FindSequentialOut(
    params_in.p, params_in.t, params_in.q,  // рабочие параметры газового потока на входе
    composition.density_std_cond, composition.co2, composition.n2, // состав газа
    passport.d_inner_, passport.d_outer_, passport.roughness_coeff_, passport.hydraulic_efficiency_coeff_, // св-ва трубы
    passport.t_env_, passport.heat_exchange_coeff_, // св-ва внешней среды (тоже входят в пасспорт трубы)
    length_of_segment, number_of_segments, // длина сегмента и кол-во сегментов
    &t_out, &p_out); 

  float eps = 1.0e-4;
  ASSERT_LE(abs(p_out - 2.9721224), eps);
  ASSERT_LE(abs(t_out - 280.14999), eps);
}

 int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}


// От London
/*
// Testing and mocking
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <functional>
#include <list>

// для getch()
#include "conio.h"

// London
#include "gas.h"
#include "gas_count_functions.cuh"

#include "passport_pipe.h"
#include "manager_edge_model_pipe_sequential.h"
#include "model_pipe_sequential.h"
#include "loader_sardan.h"
#include "model_pipe_sequential_functions_proto.cuh"
#include "manager_edge_model_pipe_sequential_cuda.cuh"

#include "utils.h"

TEST(PipeSequential, CountSequentialOut)
{
	// Тест расчёта трубы методом последовательного счёта
	// Тестировать будем так - зададим входные данные, получим расчётные рез-ты
	// и сравним их с рез-ми Весты для таких же входных данных.
	// Если результаты окажутся похожими, сохраним их и в дальнейшем будем
	// использовать в качестве эталона.

	// Задаём пасспортные свойства трубы.
 	PassportPipe passport;
	FillTestPassportPipe(&passport);

	// Задаём свойства газа на входе.
	GasCompositionReduced composition;
	GasWorkParameters params_in;
	composition.density_std_cond = 0.6865365; // [кг/м3]
	composition.co2 = 0;
	composition.n2 = 0;
	params_in.p = 5; // [МПа]
	params_in.t = 293.15; // [К]
	params_in.q = 387.843655734; // [м3/сек]

	// Производим расчёт параметров газа на выходе.
	float p_out;
	float t_out;

	float number_of_segments = 10;
	float length_of_segment = passport.length_ / number_of_segments;
	// Получены результаты p_out = 2.9721224, t_out = 280.14999 (В Весте - p_out = 2.9769, t_out = 279.78)
	// Результаты очень похожи на Весту, принимаем их за эталон.
	// Видна особенность - у меня температура выходит на Тос, а в Весте - продолжает падать.
	// ToDo: разобраться точно с этой особенностью.
	FindSequentialOut(
		params_in.p, params_in.t, params_in.q,  // рабочие параметры газового потока на входе
		composition.density_std_cond, composition.co2, composition.n2, // состав газа
		passport.d_inner_, passport.d_outer_, passport.roughness_coeff_, passport.hydraulic_efficiency_coeff_, // св-ва трубы
		passport.t_env_, passport.heat_exchange_coeff_, // св-ва внешней среды (тоже входят в пасспорт трубы)
		length_of_segment, number_of_segments, // длина сегмента и кол-во сегментов
		&t_out, &p_out); 

	float eps = 1.0e-4;
	ASSERT_LE(abs(p_out - 2.9721224), eps);
	ASSERT_LE(abs(t_out - 280.14999), eps);
}

TEST(PipeSequential, Count)
{
	// Тестируем расчёт трубы.
	// Методика тестирования поянтна - сравнить с Вестой, если похоже - 
	// сохранить полученный результат и принять за эталон.
	// Задаём пасспортные свойства трубы.
 	PassportPipe passport;
	FillTestPassportPipe(&passport);

	// Задаём свойства газа на входе.
	Gas gas_in;
	FillTestGasIn(&gas_in);

	// задаём параметры газа на выходе
	Gas gas_out;
	FillTestGasOut(&gas_out);

	// Создаём объект трубы
	ModelPipeSequential pipe(&passport);
	pipe.set_gas_in(&gas_in);
	pipe.set_gas_out(&gas_out);
	pipe.Count();

	// Результат Весты для данной конфигурации - q = 387.84
	// результат моего расчёта - 385.8383, что похоже на Весту, но не совпадает.
	// Сохраняем это решение в качестве эталонного - при дальнейших изменениях, тест
	// будет сигнализировать, что что-то изменилось, будем смотреть и разбираться.
	float eps = 0.1;
	ASSERT_LE(abs(pipe.q() - 385.83), eps);
}

TEST(Cuda, Cuda)
{
	PassportPipe passport;
	FillTestPassportPipe(&passport);
	Gas gas_in;
	FillTestGasIn(&gas_in);
	Gas gas_out;
	FillTestGasOut(&gas_out);

	ManagerEdgeModelPipeSequentialCuda manager_cuda;

	std::vector<Edge*> edges;
	edges.resize(4);

	for(int i = 0; i < 4; i++)
	{
		edges[i] = manager_cuda.CreateEdge(&passport);
	}

	std::for_each(edges.begin(), edges.end(), [gas_in, gas_out](Edge* edge)
	{
		edge->set_gas_in(&gas_in);
		edge->set_gas_out(&gas_out);
	});

	// Отлаживаемый вариант. Можно ли отлаживать лямбды, интересно?
	//edges[0]->set_gas_in(&gas_in);
	//edges[0]->set_gas_out(&gas_out);

	//edges[1]->set_gas_in(&gas_in);
	//edges[1]->set_gas_out(&gas_out);

	manager_cuda.CountAll();
}


//TEST(ManagerEdgeModelPipeSequential, LoadTest)
//{
//	// Создадим в менеджере 50 000 труб, и рассчитаем их 100 раз
//	// Подготовим трубу для расчёта
//	PassportPipe &passport;
//	FillTestPassportPipe(PassportPipe* passport)

//	// Задаём свойства газа на входе.
//	GasCompositionReduced composition;
//	GasWorkParameters params_in;
//	composition.density_std_cond = 0.6865365; // [кг/м3]
//	composition.co2 = 0;
//	composition.n2 = 0;
//	params_in.p = 5; // [МПа]
//	params_in.t = 293.15; // [К]
//	params_in.q = 387.843655734; // [м3/сек]
//	Gas gas_in;
//	gas_in.composition = composition;
//	gas_in.work_parameters = params_in;

//	// задаём параметры газа на выходе
//	Gas gas_out = gas_in;
//	gas_out.work_parameters.p = 3;
//	gas_out.work_parameters.t = 0;
//	gas_out.work_parameters.q = 0;

//	Edge *edge;
//	ManagerEdgeModelPipeSequential manager;

//	// Заполняем менеджер объектами
//	for(int i = 50000; i > 0; --i)
//	{
//		edge = manager.CreateEdge(&passport);
//		edge->set_gas_in(&gas_in);
//		edge->set_gas_out(&gas_out);
//	}

//	// Имитируем расчёт 100 итераций
//	std::cout << "Performing 10 iterations..." << std::endl;
//	for(int i = 1; i <= 10; ++i)
//	{
//		manager.CountAll();
//		std::cout << i << " iterations performed" << std::endl;
//	}
//}


//TEST(LoaderSardan, LoaderSardan)
//{
//	loader LoaderSardan;
//	loader.Load();
//}

int main(int argc, char** argv) {
  // The following line must be executed to initialize Google Mock
  // (and Google Test) before running the tests.
  ::testing::InitGoogleMock(&argc, argv);
  
  int result = RUN_ALL_TESTS();

  // Ожидаем нажатия клавиши и возвращаем результат, с которым 
  // выполнились тесты
  std::cout << "Press any key..." << std::endl;
  _getch();
  return result;
}
Конец London
*/ 