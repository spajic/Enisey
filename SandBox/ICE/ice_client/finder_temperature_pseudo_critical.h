// **********************************************************************
//
// Copyright (c) 2003-2011 ZeroC, Inc. All rights reserved.
//
// This copy of Ice is licensed to you under the terms described in the
// ICE_LICENSE file included in this distribution.
//
// **********************************************************************
//
// Ice version 3.4.2
//
// <auto-generated>
//
// Generated from file `finder_temperature_pseudo_critical.ice'
//
// Warning: do not edit this file.
//
// </auto-generated>
//

#ifndef __C__Enisey_src_SandBox_ICE_ICE_Client_finder_temperature_pseudo_critical_h__
#define __C__Enisey_src_SandBox_ICE_ICE_Client_finder_temperature_pseudo_critical_h__

#include <Ice/LocalObjectF.h>
#include <Ice/ProxyF.h>
#include <Ice/ObjectF.h>
#include <Ice/Exception.h>
#include <Ice/LocalObject.h>
#include <Ice/Proxy.h>
#include <Ice/Object.h>
#include <Ice/Outgoing.h>
#include <Ice/OutgoingAsync.h>
#include <Ice/Incoming.h>
#include <Ice/Direct.h>
#include <IceUtil/ScopedArray.h>
#include <Ice/StreamF.h>
#include <C:/Enisey/src/SandBox/ICE/ICE_server/common_types.h>
#include <Ice/UndefSysMacros.h>

#ifndef ICE_IGNORE_VERSION
#   if ICE_INT_VERSION / 100 != 304
#       error Ice version mismatch!
#   endif
#   if ICE_INT_VERSION % 100 > 50
#       error Beta header file detected
#   endif
#   if ICE_INT_VERSION % 100 < 2
#       error Ice patch level mismatch!
#   endif
#endif

namespace IceProxy
{

namespace Enisey
{

class FinderTemperaturePseudoCritical;

}

}

namespace Enisey
{

class FinderTemperaturePseudoCritical;
bool operator==(const FinderTemperaturePseudoCritical&, const FinderTemperaturePseudoCritical&);
bool operator<(const FinderTemperaturePseudoCritical&, const FinderTemperaturePseudoCritical&);

}

namespace IceInternal
{

::Ice::Object* upCast(::Enisey::FinderTemperaturePseudoCritical*);
::IceProxy::Ice::Object* upCast(::IceProxy::Enisey::FinderTemperaturePseudoCritical*);

}

namespace Enisey
{

typedef ::IceInternal::Handle< ::Enisey::FinderTemperaturePseudoCritical> FinderTemperaturePseudoCriticalPtr;
typedef ::IceInternal::ProxyHandle< ::IceProxy::Enisey::FinderTemperaturePseudoCritical> FinderTemperaturePseudoCriticalPrx;

void __read(::IceInternal::BasicStream*, FinderTemperaturePseudoCriticalPrx&);
void __patch__FinderTemperaturePseudoCriticalPtr(void*, ::Ice::ObjectPtr&);

}

namespace Enisey
{

class Callback_FinderTemperaturePseudoCritical_Find_Base : virtual public ::IceInternal::CallbackBase { };
typedef ::IceUtil::Handle< Callback_FinderTemperaturePseudoCritical_Find_Base> Callback_FinderTemperaturePseudoCritical_FindPtr;

}

namespace IceProxy
{

namespace Enisey
{

class FinderTemperaturePseudoCritical : virtual public ::IceProxy::Ice::Object
{
public:

    void Find(const ::Enisey::NumberSequence& DensityInStandartConditions, const ::Enisey::NumberSequence& Nitrogen, const ::Enisey::NumberSequence& Hydrocarbon, ::Enisey::NumberSequence& TemperaturePseudoCritical)
    {
        Find(DensityInStandartConditions, Nitrogen, Hydrocarbon, TemperaturePseudoCritical, 0);
    }
    void Find(const ::Enisey::NumberSequence& DensityInStandartConditions, const ::Enisey::NumberSequence& Nitrogen, const ::Enisey::NumberSequence& Hydrocarbon, ::Enisey::NumberSequence& TemperaturePseudoCritical, const ::Ice::Context& __ctx)
    {
        Find(DensityInStandartConditions, Nitrogen, Hydrocarbon, TemperaturePseudoCritical, &__ctx);
    }

    ::Ice::AsyncResultPtr begin_Find(const ::Enisey::NumberSequence& DensityInStandartConditions, const ::Enisey::NumberSequence& Nitrogen, const ::Enisey::NumberSequence& Hydrocarbon)
    {
        return begin_Find(DensityInStandartConditions, Nitrogen, Hydrocarbon, 0, ::IceInternal::__dummyCallback, 0);
    }

    ::Ice::AsyncResultPtr begin_Find(const ::Enisey::NumberSequence& DensityInStandartConditions, const ::Enisey::NumberSequence& Nitrogen, const ::Enisey::NumberSequence& Hydrocarbon, const ::Ice::Context& __ctx)
    {
        return begin_Find(DensityInStandartConditions, Nitrogen, Hydrocarbon, &__ctx, ::IceInternal::__dummyCallback, 0);
    }

    ::Ice::AsyncResultPtr begin_Find(const ::Enisey::NumberSequence& DensityInStandartConditions, const ::Enisey::NumberSequence& Nitrogen, const ::Enisey::NumberSequence& Hydrocarbon, const ::Ice::CallbackPtr& __del, const ::Ice::LocalObjectPtr& __cookie = 0)
    {
        return begin_Find(DensityInStandartConditions, Nitrogen, Hydrocarbon, 0, __del, __cookie);
    }

    ::Ice::AsyncResultPtr begin_Find(const ::Enisey::NumberSequence& DensityInStandartConditions, const ::Enisey::NumberSequence& Nitrogen, const ::Enisey::NumberSequence& Hydrocarbon, const ::Ice::Context& __ctx, const ::Ice::CallbackPtr& __del, const ::Ice::LocalObjectPtr& __cookie = 0)
    {
        return begin_Find(DensityInStandartConditions, Nitrogen, Hydrocarbon, &__ctx, __del, __cookie);
    }

    ::Ice::AsyncResultPtr begin_Find(const ::Enisey::NumberSequence& DensityInStandartConditions, const ::Enisey::NumberSequence& Nitrogen, const ::Enisey::NumberSequence& Hydrocarbon, const ::Enisey::Callback_FinderTemperaturePseudoCritical_FindPtr& __del, const ::Ice::LocalObjectPtr& __cookie = 0)
    {
        return begin_Find(DensityInStandartConditions, Nitrogen, Hydrocarbon, 0, __del, __cookie);
    }

    ::Ice::AsyncResultPtr begin_Find(const ::Enisey::NumberSequence& DensityInStandartConditions, const ::Enisey::NumberSequence& Nitrogen, const ::Enisey::NumberSequence& Hydrocarbon, const ::Ice::Context& __ctx, const ::Enisey::Callback_FinderTemperaturePseudoCritical_FindPtr& __del, const ::Ice::LocalObjectPtr& __cookie = 0)
    {
        return begin_Find(DensityInStandartConditions, Nitrogen, Hydrocarbon, &__ctx, __del, __cookie);
    }

    void end_Find(::Enisey::NumberSequence& TemperaturePseudoCritical, const ::Ice::AsyncResultPtr&);
    
private:

    void Find(const ::Enisey::NumberSequence&, const ::Enisey::NumberSequence&, const ::Enisey::NumberSequence&, ::Enisey::NumberSequence&, const ::Ice::Context*);
    ::Ice::AsyncResultPtr begin_Find(const ::Enisey::NumberSequence&, const ::Enisey::NumberSequence&, const ::Enisey::NumberSequence&, const ::Ice::Context*, const ::IceInternal::CallbackBasePtr&, const ::Ice::LocalObjectPtr& __cookie = 0);
    
public:
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_context(const ::Ice::Context& __context) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_context(__context).get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_context(__context).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_adapterId(const std::string& __id) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_adapterId(__id).get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_adapterId(__id).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_endpoints(const ::Ice::EndpointSeq& __endpoints) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_endpoints(__endpoints).get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_endpoints(__endpoints).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_locatorCacheTimeout(int __timeout) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_locatorCacheTimeout(__timeout).get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_locatorCacheTimeout(__timeout).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_connectionCached(bool __cached) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_connectionCached(__cached).get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_connectionCached(__cached).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_endpointSelection(::Ice::EndpointSelectionType __est) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_endpointSelection(__est).get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_endpointSelection(__est).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_secure(bool __secure) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_secure(__secure).get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_secure(__secure).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_preferSecure(bool __preferSecure) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_preferSecure(__preferSecure).get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_preferSecure(__preferSecure).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_router(const ::Ice::RouterPrx& __router) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_router(__router).get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_router(__router).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_locator(const ::Ice::LocatorPrx& __locator) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_locator(__locator).get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_locator(__locator).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_collocationOptimized(bool __co) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_collocationOptimized(__co).get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_collocationOptimized(__co).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_twoway() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_twoway().get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_twoway().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_oneway() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_oneway().get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_oneway().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_batchOneway() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_batchOneway().get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_batchOneway().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_datagram() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_datagram().get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_datagram().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_batchDatagram() const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_batchDatagram().get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_batchDatagram().get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_compress(bool __compress) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_compress(__compress).get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_compress(__compress).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_timeout(int __timeout) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_timeout(__timeout).get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_timeout(__timeout).get());
    #endif
    }
    
    ::IceInternal::ProxyHandle<FinderTemperaturePseudoCritical> ice_connectionId(const std::string& __id) const
    {
    #if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
        typedef ::IceProxy::Ice::Object _Base;
        return dynamic_cast<FinderTemperaturePseudoCritical*>(_Base::ice_connectionId(__id).get());
    #else
        return dynamic_cast<FinderTemperaturePseudoCritical*>(::IceProxy::Ice::Object::ice_connectionId(__id).get());
    #endif
    }
    
    static const ::std::string& ice_staticId();

private: 

    virtual ::IceInternal::Handle< ::IceDelegateM::Ice::Object> __createDelegateM();
    virtual ::IceInternal::Handle< ::IceDelegateD::Ice::Object> __createDelegateD();
    virtual ::IceProxy::Ice::Object* __newInstance() const;
};

}

}

namespace IceDelegate
{

namespace Enisey
{

class FinderTemperaturePseudoCritical : virtual public ::IceDelegate::Ice::Object
{
public:

    virtual void Find(const ::Enisey::NumberSequence&, const ::Enisey::NumberSequence&, const ::Enisey::NumberSequence&, ::Enisey::NumberSequence&, const ::Ice::Context*) = 0;
};

}

}

namespace IceDelegateM
{

namespace Enisey
{

class FinderTemperaturePseudoCritical : virtual public ::IceDelegate::Enisey::FinderTemperaturePseudoCritical,
                                        virtual public ::IceDelegateM::Ice::Object
{
public:

    virtual void Find(const ::Enisey::NumberSequence&, const ::Enisey::NumberSequence&, const ::Enisey::NumberSequence&, ::Enisey::NumberSequence&, const ::Ice::Context*);
};

}

}

namespace IceDelegateD
{

namespace Enisey
{

class FinderTemperaturePseudoCritical : virtual public ::IceDelegate::Enisey::FinderTemperaturePseudoCritical,
                                        virtual public ::IceDelegateD::Ice::Object
{
public:

    virtual void Find(const ::Enisey::NumberSequence&, const ::Enisey::NumberSequence&, const ::Enisey::NumberSequence&, ::Enisey::NumberSequence&, const ::Ice::Context*);
};

}

}

namespace Enisey
{

class FinderTemperaturePseudoCritical : virtual public ::Ice::Object
{
public:

    typedef FinderTemperaturePseudoCriticalPrx ProxyType;
    typedef FinderTemperaturePseudoCriticalPtr PointerType;
    
    virtual ::Ice::ObjectPtr ice_clone() const;

    virtual bool ice_isA(const ::std::string&, const ::Ice::Current& = ::Ice::Current()) const;
    virtual ::std::vector< ::std::string> ice_ids(const ::Ice::Current& = ::Ice::Current()) const;
    virtual const ::std::string& ice_id(const ::Ice::Current& = ::Ice::Current()) const;
    static const ::std::string& ice_staticId();

    virtual void Find(const ::Enisey::NumberSequence&, const ::Enisey::NumberSequence&, const ::Enisey::NumberSequence&, ::Enisey::NumberSequence&, const ::Ice::Current& = ::Ice::Current()) = 0;
    ::Ice::DispatchStatus ___Find(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual ::Ice::DispatchStatus __dispatch(::IceInternal::Incoming&, const ::Ice::Current&);

    virtual void __write(::IceInternal::BasicStream*) const;
    virtual void __read(::IceInternal::BasicStream*, bool);
// COMPILERFIX: Stream API is not supported with VC++ 6
#if !defined(_MSC_VER) || (_MSC_VER >= 1300)
    virtual void __write(const ::Ice::OutputStreamPtr&) const;
    virtual void __read(const ::Ice::InputStreamPtr&, bool);
#endif
};

inline bool operator==(const FinderTemperaturePseudoCritical& l, const FinderTemperaturePseudoCritical& r)
{
    return static_cast<const ::Ice::Object&>(l) == static_cast<const ::Ice::Object&>(r);
}

inline bool operator<(const FinderTemperaturePseudoCritical& l, const FinderTemperaturePseudoCritical& r)
{
    return static_cast<const ::Ice::Object&>(l) < static_cast<const ::Ice::Object&>(r);
}

}

namespace Enisey
{

template<class T>
class CallbackNC_FinderTemperaturePseudoCritical_Find : public Callback_FinderTemperaturePseudoCritical_Find_Base, public ::IceInternal::TwowayCallbackNC<T>
{
public:

    typedef IceUtil::Handle<T> TPtr;

    typedef void (T::*Exception)(const ::Ice::Exception&);
    typedef void (T::*Sent)(bool);
    typedef void (T::*Response)(const ::Enisey::NumberSequence&);

    CallbackNC_FinderTemperaturePseudoCritical_Find(const TPtr& obj, Response cb, Exception excb, Sent sentcb)
        : ::IceInternal::TwowayCallbackNC<T>(obj, cb != 0, excb, sentcb), response(cb)
    {
    }

    virtual void __completed(const ::Ice::AsyncResultPtr& __result) const
    {
        ::Enisey::FinderTemperaturePseudoCriticalPrx __proxy = ::Enisey::FinderTemperaturePseudoCriticalPrx::uncheckedCast(__result->getProxy());
        ::Enisey::NumberSequence TemperaturePseudoCritical;
        try
        {
            __proxy->end_Find(TemperaturePseudoCritical, __result);
        }
        catch(::Ice::Exception& ex)
        {
#if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
            __exception(__result, ex);
#else
            ::IceInternal::CallbackNC<T>::__exception(__result, ex);
#endif
            return;
        }
        if(response)
        {
#if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
            (callback.get()->*response)(TemperaturePseudoCritical);
#else
            (::IceInternal::CallbackNC<T>::callback.get()->*response)(TemperaturePseudoCritical);
#endif
        }
    }

    Response response;
};

template<class T> Callback_FinderTemperaturePseudoCritical_FindPtr
newCallback_FinderTemperaturePseudoCritical_Find(const IceUtil::Handle<T>& instance, void (T::*cb)(const ::Enisey::NumberSequence&), void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_FinderTemperaturePseudoCritical_Find<T>(instance, cb, excb, sentcb);
}

template<class T> Callback_FinderTemperaturePseudoCritical_FindPtr
newCallback_FinderTemperaturePseudoCritical_Find(T* instance, void (T::*cb)(const ::Enisey::NumberSequence&), void (T::*excb)(const ::Ice::Exception&), void (T::*sentcb)(bool) = 0)
{
    return new CallbackNC_FinderTemperaturePseudoCritical_Find<T>(instance, cb, excb, sentcb);
}

template<class T, typename CT>
class Callback_FinderTemperaturePseudoCritical_Find : public Callback_FinderTemperaturePseudoCritical_Find_Base, public ::IceInternal::TwowayCallback<T, CT>
{
public:

    typedef IceUtil::Handle<T> TPtr;

    typedef void (T::*Exception)(const ::Ice::Exception& , const CT&);
    typedef void (T::*Sent)(bool , const CT&);
    typedef void (T::*Response)(const ::Enisey::NumberSequence&, const CT&);

    Callback_FinderTemperaturePseudoCritical_Find(const TPtr& obj, Response cb, Exception excb, Sent sentcb)
        : ::IceInternal::TwowayCallback<T, CT>(obj, cb != 0, excb, sentcb), response(cb)
    {
    }

    virtual void __completed(const ::Ice::AsyncResultPtr& __result) const
    {
        ::Enisey::FinderTemperaturePseudoCriticalPrx __proxy = ::Enisey::FinderTemperaturePseudoCriticalPrx::uncheckedCast(__result->getProxy());
        ::Enisey::NumberSequence TemperaturePseudoCritical;
        try
        {
            __proxy->end_Find(TemperaturePseudoCritical, __result);
        }
        catch(::Ice::Exception& ex)
        {
#if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
            __exception(__result, ex);
#else
            ::IceInternal::Callback<T, CT>::__exception(__result, ex);
#endif
            return;
        }
        if(response)
        {
#if defined(_MSC_VER) && (_MSC_VER < 1300) // VC++ 6 compiler bug
            (callback.get()->*response)(TemperaturePseudoCritical, CT::dynamicCast(__result->getCookie()));
#else
            (::IceInternal::Callback<T, CT>::callback.get()->*response)(TemperaturePseudoCritical, CT::dynamicCast(__result->getCookie()));
#endif
        }
    }

    Response response;
};

template<class T, typename CT> Callback_FinderTemperaturePseudoCritical_FindPtr
newCallback_FinderTemperaturePseudoCritical_Find(const IceUtil::Handle<T>& instance, void (T::*cb)(const ::Enisey::NumberSequence&, const CT&), void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_FinderTemperaturePseudoCritical_Find<T, CT>(instance, cb, excb, sentcb);
}

template<class T, typename CT> Callback_FinderTemperaturePseudoCritical_FindPtr
newCallback_FinderTemperaturePseudoCritical_Find(T* instance, void (T::*cb)(const ::Enisey::NumberSequence&, const CT&), void (T::*excb)(const ::Ice::Exception&, const CT&), void (T::*sentcb)(bool, const CT&) = 0)
{
    return new Callback_FinderTemperaturePseudoCritical_Find<T, CT>(instance, cb, excb, sentcb);
}

}

#endif
