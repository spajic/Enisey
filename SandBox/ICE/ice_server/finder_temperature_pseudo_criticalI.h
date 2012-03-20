#ifndef __C__Enisey_src_SandBox_ICE_ICE_server_finder_temperature_pseudo_criticalI_h__
#define __C__Enisey_src_SandBox_ICE_ICE_server_finder_temperature_pseudo_criticalI_h__

#include <finder_temperature_pseudo_critical.h>

namespace Enisey
{

class FinderTemperaturePseudoCriticalI : 
	virtual public FinderTemperaturePseudoCritical {
 public:
  virtual void Find(
      const ::Enisey::NumberSequence&,
      const ::Enisey::NumberSequence&,
      const ::Enisey::NumberSequence&,
      ::Enisey::NumberSequence&,
      const Ice::Current&);
  // Добавление себя в ASM (ActiveServantsMap переданного адаптера.
  void Activate(const Ice::ObjectAdapterPtr& adapter);
};

typedef IceUtil::Handle<FinderTemperaturePseudoCriticalI> 
	FinderTemperaturePseudoCriticalIPtr;
}

#endif
