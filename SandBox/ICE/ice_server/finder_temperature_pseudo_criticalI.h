#ifndef __C__Enisey_src_SandBox_ICE_ICE_server_finder_temperature_pseudo_criticalI_h__
#define __C__Enisey_src_SandBox_ICE_ICE_server_finder_temperature_pseudo_criticalI_h__

#include <finder_temperature_pseudo_critical.h>

namespace Enisey
{

class FinderTemperaturePseudoCriticalI : 
	virtual public FinderTemperaturePseudoCritical {
 public:
  virtual void Find(
      const ::Enisey::NumberSequence& DensityInStandartConditions,
      const ::Enisey::NumberSequence& Nitrogen,
      const ::Enisey::NumberSequence& Hydrocarbon,
      ::Enisey::NumberSequence& TemperaturePseudoCritical,
      const Ice::Current&);
  // ���������� ���� � ASM (ActiveServantsMap ����������� ��������.
  void Activate(const Ice::ObjectAdapterPtr& adapter);
};

typedef IceUtil::Handle<FinderTemperaturePseudoCriticalI> 
	FinderTemperaturePseudoCriticalIPtr;
}

#endif
