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

// Complex types used in this pass
typedef std::pair<BasicBlock *, Value *> StackEntry;
typedef SmallVector<StackEntry, 16> StackVector;

// Intrinsic names the UAV is annotated with
static const char *const getUavDescIntrinsic = "llvm.SI.m2s.get.uav.desc";
static const char *const pacUavDescIntrinsic = "llvm.SI.m2s.pac.uav.desc";

class SIM2SAnnotateUAV : public FunctionPass {

  static char ID;

  // Return or parameter types used in intrinsic functions
  Type *Int32;
  VectorType *Vector4Int32;

  // Intrinsic function constant
  Constant *GetUavDesc;
  Constant *PacUavDesc;

  // We can only erase after traverse all basic blocks
  SmallVector<Instruction *, 4> InstsToErase;

public:
  SIM2SAnnotateUAV() : FunctionPass(ID) {}

  bool doInitialization(Module &M) override;

  bool runOnFunction(Function &F) override;

  const char *getPassName() const override { return "SI M2S annotate UAV"; }
};

} // end anonymous namespace

char SIM2SAnnotateUAV::ID = 0;

/// \brief Initialize all the types and constants used in the pass
bool SIM2SAnnotateUAV::doInitialization(Module &M) {
  LLVMContext &Context = M.getContext();

  Int32 = Type::getInt32Ty(Context);
  Vector4Int32 = VectorType::get(Int32, 4);

  GetUavDesc = M.getOrInsertFunction(getUavDescIntrinsic, Vector4Int32, Int32,
                                     Int32, (Type *)nullptr);
  PacUavDesc = M.getOrInsertFunction(pacUavDescIntrinsic, Int32, Int32,
                                     Vector4Int32, (Type *)nullptr);
  return false;
}

/// \brief Annotate the getElementPtr with M2S UAV intrinsics
bool SIM2SAnnotateUAV::runOnFunction(Function &F) {

  std::map<std::string, CallInst *> UAVMap;

  // EntryBlock: get UAV buffer descriptors
  BasicBlock &EntryBlock = F.getEntryBlock();
  auto &Args = F.getArgumentList();
  for (auto &Arg : Args) {
    // Get argument number and address space for get.uav intrinsic function
    unsigned ArgNum = Arg.getArgNo();
    auto *ArgType =
        dyn_cast<PointerType>(F.getFunctionType()->getParamType(ArgNum));
    unsigned ArgAddrSpace = ArgType->getAddressSpace();

    switch (ArgAddrSpace) {
    // __global/__constant read from imm_const_buffer_0
    case AMDGPUAS::GLOBAL_ADDRESS:
    case AMDGPUAS::CONSTANT_ADDRESS: {
      Value *CallInstArgs[] = {ConstantInt::get(Int32, ArgAddrSpace),
                               ConstantInt::get(Int32, ArgNum)};
      CallInst *CallGetUAV =
          CallInst::Create(GetUavDesc, CallInstArgs, "uav." + Arg.getName(),
                           EntryBlock.getFirstInsertionPt());
      if (!CallGetUAV)
        return false;

      // Store in UAV map for later lookup
      UAVMap[CallGetUAV->getName().str()] = CallGetUAV;
      break;
    }
    // Others read from imm_const_buffer_1
    default: {
      // TODO: not implemented yet
      return false;
      break;
    }
    }
  }

  // All basic blocks: need to pack UAV buffer descriptor
  for (auto &BB : F.getBasicBlockList()) {
    for (BasicBlock::iterator i = BB.begin(), e = BB.end(); i != e; ++i)
      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(&*i)) {
        // Get GEP pointer operand name and search in the UAV map
        Value *GEPPtr = GEP->getPointerOperand();
        std::string UAVNameGEPPtr = "uav." + GEPPtr->getName().str();
        if (UAVMap.find(UAVNameGEPPtr) == UAVMap.end())
          return false;

        // Create CallInst and insert before GEP
        Value *GEPIdx = *(GEP->idx_begin());
        Value *Arg0 = GEPIdx;
        Value *Arg1 = UAVMap[UAVNameGEPPtr];
        Value *Args[] = {Arg0, Arg1};
        CallInst *CallPacUAV =
            CallInst::Create(PacUavDesc, Args, "pac." + GEPIdx->getName(), GEP);
        if (!CallPacUAV)
          return false;

        // Create NewGEP and replaces old GEP
        Value *Indices[] = {CallPacUAV};
        GetElementPtrInst *NewGEP = GetElementPtrInst::Create(
            GEP->getSourceElementType(), GEPPtr, Indices, GEP->getName(), GEP);
        NewGEP->setIsInBounds(GEP->isInBounds());
        GEP->replaceAllUsesWith(NewGEP);
        // Mark as to be erased
        InstsToErase.push_back(GEP);
      }
  }

  // We are done, remove those instructions in erase list
  for (unsigned i = 0; i < InstsToErase.size(); ++i) {
    InstsToErase[i]->eraseFromParent();
  }

  return true;
}

/// \brief Create the annotation pass
FunctionPass *llvm::createSIM2SAnnotateUAVPass() {
  return new SIM2SAnnotateUAV();
}
