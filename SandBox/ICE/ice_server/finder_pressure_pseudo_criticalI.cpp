#include <finder_pressure_pseudo_criticalI.h>
#include <Ice/Ice.h>

void Enisey::FinderPressurePseudoCriticalI::Find(
    const ::Enisey::NumberSequence& DensityInStandartConditions,
    const ::Enisey::NumberSequence& Nitrogen,
    const ::Enisey::NumberSequence& Hydrocarbon,
    ::Enisey::NumberSequence& PressurePseudoCritical,
    const Ice::Current& current) {
  // ѕока сделаем тривиальную реализацию. —мотрим length векторов, возвращаем
  // вектор соответствующего размера, заполненный значанием 66.
  std::cout << "Server for Pressure Called!\n";
  PressurePseudoCritical.resize(DensityInStandartConditions.size());
  for(unsigned int i = 0; i < DensityInStandartConditions.size(); ++i) {
    PressurePseudoCritical[i] = 66;
  }
  return;
}

void Enisey::FinderPressurePseudoCriticalI::Activate(
	const Ice::ObjectAdapterPtr& adapter) {
  Ice::Identity id;
  // ѕока используем identity = 'Pressure' - этот идентификатор потребуетс€
  // клиенту.
  id.name = "Pressure"; 
  try {
	adapter->add(this, id);
  }
  catch(const Ice::Exception& ex) {
    std::cout << ex.what();
  }
}