//===----------------------- SIM2SAnnotateUAV.cpp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Annotates the getElementPtr with M2S specific intrinsics.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"

using namespace llvm;

#define DEBUG_TYPE "si-m2s-annotate-uav"

namespace {

// Intrinsic names the UAV is annotated with
static const char *const getUavDescIntrinsic = "llvm.SI.m2s.get.uav.desc";
static const char *const pacUavDescI8Intrinsic =
    "llvm.SI.m2s.pac.uav.desc.i8.global";
static const char *const pacUavDescI32Intrinsic =
    "llvm.SI.m2s.pac.uav.desc.i32.global";
static const char *const pacUavDescV4I8GlobalIntrinsic =
    "llvm.SI.m2s.pac.uav.desc.v4i8.global";
static const char *const pacUavDescV4I32GlobalIntrinsic =
    "llvm.SI.m2s.pac.uav.desc.v4i32.global";
static const char *const pacUavDescFloatGlobalIntrinsic =
    "llvm.SI.m2s.pac.uav.desc.float.global";
static const char *const pacUavDescV4FloatGlobalIntrinsic =
    "llvm.SI.m2s.pac.uav.desc.v4float.global";

class SIM2SAnnotateUAV : public FunctionPass {

  static char ID;

  Module *M;

  // Return or parameter types used in intrinsic functions
  Type *Int32;
  VectorType *Vector4Int32;

  // Intrinsic function constant
  Constant *GetUavDesc;
  Constant *PacUavDesc;

  // We can only erase after traverse all basic blocks
  std::vector<Instruction *> InstsToErase;

  Constant *getPacUavFunc(Value *Arg);

public:
  SIM2SAnnotateUAV() : FunctionPass(ID) {}

  bool doInitialization(Module &M) override;

  bool runOnFunction(Function &F) override;

  const char *getPassName() const override { return "SI M2S annotate UAV"; }
};

} // end anonymous namespace

char SIM2SAnnotateUAV::ID = 0;

Constant *SIM2SAnnotateUAV::getPacUavFunc(Value *Arg) {
  Type *PtrType = Arg->getType();
  Type *PtrElemType = PtrType->getPointerElementType();

  // TODO: more memory addresses
  if (PtrElemType->isVectorTy()) {
    Type *VectorElemType = PtrElemType->getVectorElementType();
    unsigned numElems = PtrElemType->getVectorNumElements();

    switch (numElems) {
    default:
      break;
    case 4:
      if (VectorElemType->isIntegerTy(8))
        return M->getOrInsertFunction(pacUavDescI8Intrinsic, PtrType,
                                      PtrType, Vector4Int32, (Type *)nullptr);
      if (VectorElemType->isIntegerTy(32))
        return M->getOrInsertFunction(pacUavDescV4I32GlobalIntrinsic, PtrType,
                                      PtrType, Vector4Int32, (Type *)nullptr);
      else if (VectorElemType->isFloatingPointTy())
        return M->getOrInsertFunction(pacUavDescV4FloatGlobalIntrinsic, PtrType,
                                      PtrType, Vector4Int32, (Type *)nullptr);
      break;
    }
  } else {
    if (PtrElemType->isIntegerTy())
      return M->getOrInsertFunction(pacUavDescI32Intrinsic, PtrType, PtrType,
                                    Vector4Int32, (Type *)nullptr);
    else if (PtrElemType->isFloatingPointTy())
      return M->getOrInsertFunction(pacUavDescFloatGlobalIntrinsic, PtrType,
                                    PtrType, Vector4Int32, (Type *)nullptr);
  }

  return nullptr;
}

/// \brief Initialize all the types and constants used in the pass
bool SIM2SAnnotateUAV::doInitialization(Module &M) {
  LLVMContext &Context = M.getContext();
  this->M = &M;
  Int32 = Type::getInt32Ty(Context);
  Vector4Int32 = VectorType::get(Int32, 4);

  GetUavDesc = M.getOrInsertFunction(getUavDescIntrinsic, Vector4Int32, Int32,
                                     Int32, (Type *)nullptr);
  return false;
}

/// \brief Annotate the getElementPtr with M2S UAV intrinsics
bool SIM2SAnnotateUAV::runOnFunction(Function &F) {

  std::map<Value *, CallInst *> UAVMap;
  UAVMap.clear();

  // EntryBlock: get UAV buffer descriptors
  BasicBlock &EntryBlock = F.getEntryBlock();
  auto &Args = F.getArgumentList();
  unsigned uavCount = 0;
  for (auto &Arg : Args) {
    // Get argument index and address space for get.uav intrinsic function
    unsigned ArgIdx = Arg.getArgNo();

    auto *ArgType =
        dyn_cast<PointerType>(F.getFunctionType()->getParamType(ArgIdx));
    if (ArgType) {
      unsigned ArgAddrSpace = ArgType->getAddressSpace();

      switch (ArgAddrSpace) {
      // __global/__constant read from imm_const_buffer_0
      case AMDGPUAS::GLOBAL_ADDRESS:
      case AMDGPUAS::CONSTANT_ADDRESS: {
        Value *CallInstArgs[] = {ConstantInt::get(Int32, ArgAddrSpace),
                                 ConstantInt::get(Int32, 0x50 + uavCount * 8)};
        uavCount++;
        CallInst *CallGetUAV =
            CallInst::Create(GetUavDesc, CallInstArgs, "uav." + Arg.getName(),
                             EntryBlock.getFirstInsertionPt());
        if (!CallGetUAV)
          return false;

        // Store in UAV map for later lookup
        UAVMap[&Arg] = CallGetUAV;
        break;
      }
      // Others read from imm_const_buffer_1
      default: {
        // TODO: not implemented yet
        break;
      }
      }
    }
  }

  // All basic blocks: need to pack UAV buffer descriptor
  // The goal is to find corresponding llvm.SI.m2s.get.uav.desc instruction. So
  // we can pack it to the operand of LD instruction, which will be used in
  // SelectMTBUFOffsetM2S complex pattern
  for (auto &BB : F.getBasicBlockList()) {
    for (BasicBlock::iterator i = BB.begin(), e = BB.end(); i != e; ++i) {
      if (LoadInst *LD = dyn_cast<LoadInst>(&*i)) {
        // Local address space doesn't need to use UAV
        if (LD->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS)
          continue;

        // Get LD pointer operand and find CallInst mapping
        Value *LDPtrOp = LD->getPointerOperand();

        // Args for llvm.SI.m2s.pac.uav.desc
        Value *Arg0 = LDPtrOp;
        Value *Arg1 = nullptr;

        // First check if LD pointer operand is coming directly from parameter
        if (UAVMap.find(LDPtrOp) != UAVMap.end())
          Arg1 = UAVMap[LDPtrOp];
        else {
          // Other possibilities: GEP/BitCast/PhiNode/IntToPtr
          auto *GEP = dyn_cast<GetElementPtrInst>(LDPtrOp);
          auto *BC = dyn_cast<BitCastInst>(LDPtrOp);
          auto *PN = dyn_cast<PHINode>(LDPtrOp);
          auto *ITP = dyn_cast<IntToPtrInst>(LDPtrOp);
          while (!Arg1 && (GEP || BC || PN || ITP)) {
            if (BC) {
              GEP = dyn_cast<GetElementPtrInst>(BC->getOperand(0));
              PN = dyn_cast<PHINode>(BC->getOperand(0));
              ITP = dyn_cast<IntToPtrInst>(BC->getOperand(0));
              BC = dyn_cast<BitCastInst>(BC->getOperand(0));
            } else if (PN) {
              if (UAVMap.find(PN->getIncomingValue(1)) != UAVMap.end())
                Arg1 = UAVMap[PN->getIncomingValue(1)];
              GEP = dyn_cast<GetElementPtrInst>(PN->getIncomingValue(1));
              BC = dyn_cast<BitCastInst>(PN->getIncomingValue(1));
              ITP = dyn_cast<IntToPtrInst>(PN->getIncomingValue(1));
              PN = dyn_cast<PHINode>(PN->getIncomingValue(1));
            } else if (GEP) {
              auto GEPOp = GEP->getPointerOperand();
              if (UAVMap.find(GEPOp) != UAVMap.end())
                Arg1 = UAVMap[GEPOp];
              else {
                PN = dyn_cast<PHINode>(GEPOp);
                BC = dyn_cast<BitCastInst>(GEPOp);
                ITP = dyn_cast<IntToPtrInst>(GEPOp);
                GEP = dyn_cast<GetElementPtrInst>(GEPOp);
              }
            } else if (ITP) {
              GEP = dyn_cast<GetElementPtrInst>(ITP->getOperand(0));
              PN = dyn_cast<PHINode>(ITP->getOperand(0));
              BC = dyn_cast<BitCastInst>(ITP->getOperand(0));
              ITP = dyn_cast<IntToPtrInst>(ITP->getOperand(0));
            }
          }
        }

        // Create CallInst and insert before LD
        assert(Arg1 && "Arg1 = nullptr");
        Value *Args[] = {Arg0, Arg1};
        PacUavDesc = getPacUavFunc(Arg0);
        assert(PacUavDesc && "PacUavDesc = nullptr");
        std::string CallInstName = Arg0->hasName()
                                       ? Arg0->getName().str()
                                       : std::to_string(Arg0->getValueID());
        CallInst *CallPacUAV =
            CallInst::Create(PacUavDesc, Args, "pac." + CallInstName, LD);
        if (!CallPacUAV)
          return false;

        // Create New LD and replaces old LD
        LoadInst *NewLD = new LoadInst(CallPacUAV, LD->getName(), LD);
        LD->replaceAllUsesWith(NewLD);
        // Mark as to be erased
        InstsToErase.push_back(LD);
      } else if (StoreInst *ST = dyn_cast<StoreInst>(&*i)) {
        // Local address space doesn't need to use UAV
        if (ST->getPointerAddressSpace() == AMDGPUAS::LOCAL_ADDRESS)
          continue;

        // Get ST pointer operand and find CallInst mapping
        Value *STPtrOp = ST->getPointerOperand();

        // Args for llvm.SI.m2s.pac.uav.desc
        Value *Arg0 = STPtrOp;
        Value *Arg1 = nullptr;

        // First check if ST pointer operand is coming directly from parameter
        if (UAVMap.find(STPtrOp) != UAVMap.end())
          Arg1 = UAVMap[STPtrOp];
        else {
          // Other possibilities: GEP/BitCast/PhiNode
          auto *GEP = dyn_cast<GetElementPtrInst>(STPtrOp);
          auto *BC = dyn_cast<BitCastInst>(STPtrOp);
          auto *PN = dyn_cast<PHINode>(STPtrOp);
          auto *ITP = dyn_cast<IntToPtrInst>(STPtrOp);
          while (!Arg1 && (GEP || BC || PN || ITP)) {
            if (BC) {
              GEP = dyn_cast<GetElementPtrInst>(BC->getOperand(0));
              PN = dyn_cast<PHINode>(BC->getOperand(0));
              ITP = dyn_cast<IntToPtrInst>(BC->getOperand(0));
              BC = dyn_cast<BitCastInst>(BC->getOperand(0));
            } else if (PN) {
              if (UAVMap.find(PN->getIncomingValue(1)) != UAVMap.end())
                Arg1 = UAVMap[PN->getIncomingValue(1)];
              GEP = dyn_cast<GetElementPtrInst>(PN->getIncomingValue(1));
              BC = dyn_cast<BitCastInst>(PN->getIncomingValue(1));
              ITP = dyn_cast<IntToPtrInst>(PN->getIncomingValue(1));
              PN = dyn_cast<PHINode>(PN->getIncomingValue(1));
            } else if (GEP) {
              auto GEPOp = GEP->getPointerOperand();
              if (UAVMap.find(GEPOp) != UAVMap.end())
                Arg1 = UAVMap[GEPOp];
              else {
                PN = dyn_cast<PHINode>(GEPOp);
                BC = dyn_cast<BitCastInst>(GEPOp);
                ITP = dyn_cast<IntToPtrInst>(GEPOp);
                GEP = dyn_cast<GetElementPtrInst>(GEPOp);
              }
            } else if (ITP) {
              GEP = dyn_cast<GetElementPtrInst>(ITP->getOperand(0));
              PN = dyn_cast<PHINode>(ITP->getOperand(0));
              BC = dyn_cast<BitCastInst>(ITP->getOperand(0));
              ITP = dyn_cast<IntToPtrInst>(ITP->getOperand(0));
            }
          }
        }

        // Create CallInst and insert before ST
        assert(Arg1 && "Arg1 = nullptr");
        Value *Args[] = {Arg0, Arg1};
        PacUavDesc = getPacUavFunc(Arg0);
        assert(PacUavDesc && "PacUavDesc = nullptr");
        std::string CallInstName = Arg0->hasName()
                                       ? Arg0->getName().str()
                                       : std::to_string(Arg0->getValueID());
        CallInst *CallPacUAV =
            CallInst::Create(PacUavDesc, Args, "pac." + CallInstName, ST);
        if (!CallPacUAV)
          return false;

        // Create New ST and replaces old ST
        StoreInst *NewST = new StoreInst(ST->getValueOperand(), CallPacUAV, ST);
        ST->replaceAllUsesWith(NewST);
        // Mark as to be erased
        InstsToErase.push_back(ST);
      }
    }
  }

  // We are done, remove those instructions in erase list
  for (auto &Inst : InstsToErase)
    Inst->eraseFromParent();
  InstsToErase.clear();

  return true;
}

/// \brief Create the annotation pass
FunctionPass *llvm::createSIM2SAnnotateUAVPass() {
  return new SIM2SAnnotateUAV();
}
