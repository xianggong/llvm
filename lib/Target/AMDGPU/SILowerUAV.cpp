//===------- SILowerUAV.cpp - Lower UAV intrinsics for Multi2Sim ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
/// \brief This pass lowers the pseudo UAV instructions to real machine
/// instructions.
///
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/Constants.h"

using namespace llvm;

namespace {

class SILowerUAVPass : public MachineFunctionPass {

private:
  static char ID;
  MachineRegisterInfo *MRI;
  const SIRegisterInfo *TRI;
  const SIInstrInfo *TII;

  void LowerGetUAV(MachineInstr &MI, unsigned PtrUavTableRegs);
  void LowerPacUAV(MachineInstr &MI);

public:
  SILowerUAVPass(TargetMachine &tm)
      : MachineFunctionPass(ID), TRI(nullptr), TII(nullptr) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  const char *getPassName() const override {
    return "SI Lower M2S UAV instructions";
  }
};

} // End anonymous namespace

char SILowerUAVPass::ID = 0;

FunctionPass *llvm::createSILowerUAVPass(TargetMachine &tm) {
  return new SILowerUAVPass(tm);
}

void SILowerUAVPass::LowerGetUAV(MachineInstr &MI, unsigned PtrUavTableRegs) {
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = MI.getDebugLoc();

  // Add PtrUAVTable LiveIn and a COPY to define virtual register, otherwise it
  // is used before def and causes problem
  MBB.addLiveIn(AMDGPU::SGPR2_SGPR3);

  // Create def only when it's not defined
  auto VRegPtrUavTable = MRI->getLiveInVirtReg(PtrUavTableRegs);
  if (!MRI->getVRegDef(VRegPtrUavTable))
    BuildMI(MBB, &MI, DL, TII->get(TargetOpcode::COPY), VRegPtrUavTable)
        .addReg(AMDGPU::SGPR2_SGPR3);

  unsigned AddrSpace = MI.getOperand(1).getReg();
  unsigned Idx = MI.getOperand(2).getReg();
  MachineInstr *movAddrSpaceToReg = MRI->getVRegDef(AddrSpace);
  MachineInstr *movIdxToReg = MRI->getVRegDef(Idx);

  // Get the index of the parameter then remove these machine instructions
  unsigned IdxImm = 80;
  if (movAddrSpaceToReg)
    movAddrSpaceToReg->eraseFromParent();
  if (movIdxToReg) {
    unsigned IdxImm = movIdxToReg->getOperand(1).getImm();
    IdxImm += IdxImm * 8;
    // movIdxToReg->eraseFromParent();
  }
  // Lower it to s_load_dwordx4_imm
  unsigned Dst = MI.getOperand(0).getReg();
  BuildMI(MBB, &MI, DL, TII->get(AMDGPU::S_LOAD_DWORDX4_IMM), Dst)
      .addReg(MRI->getLiveInVirtReg(PtrUavTableRegs))
      .addImm(IdxImm);

  // Cleanup work
  MI.eraseFromParent();
}

void SILowerUAVPass::LowerPacUAV(MachineInstr &MI) {
  // Clean dummy node after instruction selection, we only used it for passing
  // UAV desc registers
  MI.eraseFromParent();
}

bool SILowerUAVPass::runOnMachineFunction(MachineFunction &MF) {
  TII = static_cast<const SIInstrInfo *>(MF.getSubtarget().getInstrInfo());
  TRI =
      static_cast<const SIRegisterInfo *>(MF.getSubtarget().getRegisterInfo());
  MRI = &MF.getRegInfo();
  unsigned PtrUavTable =
      TRI->getPreloadedValue(MF, SIRegisterInfo::PTR_UAV_TABLE);
  MF.addLiveIn(PtrUavTable, &AMDGPU::SReg_64RegClass);

  for (auto BI = MF.begin(), BE = MF.end(); BI != BE; ++BI) {
    MachineBasicBlock &MBB = *BI;
    MachineBasicBlock::iterator I, Next;
    // 1st pass, handle GET_UAV_DESC
    for (I = MBB.begin(); I != MBB.end(); I = Next) {
      Next = std::next(I);

      MachineInstr &MI = *I;
      switch (MI.getOpcode()) {
      default:
        break;
      case AMDGPU::SI_M2S_GET_UAV_DESC:
        LowerGetUAV(MI, PtrUavTable);
        break;
      }
    }
    // 2nd pass, cleanup PAC_UAV_DESC
    for (I = MBB.begin(); I != MBB.end(); I = Next) {
      Next = std::next(I);

      MachineInstr &MI = *I;
      switch (MI.getOpcode()) {
      default:
        break;
      case AMDGPU::SI_M2S_PAC_UAV_DESC:
        LowerPacUAV(MI);
        break;
      }
    }
  }

  return true;
}
