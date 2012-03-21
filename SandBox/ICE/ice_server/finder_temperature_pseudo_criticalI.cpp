#include <finder_temperature_pseudo_criticalI.h>

#include <Ice/Ice.h>
#include "finder_temperature_pseudo_critical_worker_cuda.h"

void Enisey::FinderTemperaturePseudoCriticalI::Find(
    const ::Enisey::NumberSequence& DensityInStandartConditions,
    const ::Enisey::NumberSequence& Nitrogen,
    const ::Enisey::NumberSequence& Hydrocarbon,
    ::Enisey::NumberSequence& TemperaturePseudoCritical,
    const Ice::Current& current) {
  // ���� ������� ����������� ����������. ������� length ��������, ����������
  // ������ ���������������� �������, ����������� ��������� 99.
  //for(unsigned int i = 0; i < TemperaturePseudoCritical.size(); ++i) {
  //  TemperaturePseudoCritical[i] = 99;
  //}
		std::cout << "Server Called! CALL CUDA!\n";
    FinderPressurePseudoCriticalWorkerCuda *worker_cuda = 
      new FinderPressurePseudoCriticalWorkerCuda();
    worker_cuda->Find(DensityInStandartConditions,Nitrogen, Hydrocarbon, TemperaturePseudoCritical);

    delete worker_cuda;
		//TemperaturePseudoCritical.resize(DensityInStandartConditions.size());
		//for(unsigned int i = 0; i < DensityInStandartConditions.size(); ++i) {
		//	  TemperaturePseudoCritical[i] = 99;
		//	}
  return;
}

void Enisey::FinderTemperaturePseudoCriticalI::Activate(
	const Ice::ObjectAdapterPtr& adapter) {
  Ice::Identity id;
  // ���� ���������� identity = 'Spajic' - ���� ������������� �����������
  // �������.
  id.name = "alex"; 
  try {
	adapter->add(this, id);
  }
  catch(const Ice::Exception& ex)
  {
	  std::cout << ex.what();
  }
}
