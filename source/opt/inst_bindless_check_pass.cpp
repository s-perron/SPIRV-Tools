// Copyright (c) 2017 The Khronos Group Inc.
// Copyright (c) 2017 Valve Corporation
// Copyright (c) 2017 LunarG Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "inst_bindless_check_pass.h"

static const int kSpvImageSampleImageIdInIdx = 0;
static const int kSpvSampledImageImageIdInIdx = 0;
static const int kSpvLoadPtrIdInIdx = 0;
static const int kSpvAccessChainBaseIdInIdx = 0;
static const int kSpvAccessChainIndex0IdInIdx = 0;
static const int kSpvTypePointerTypeIdInIdx = 0;
static const int kSpvTypeArrayLengthIdInIdx = 1;
static const int kSpvConstantValueInIdx = 0;

namespace spvtools {
namespace opt {

bool InstBindlessCheckPass::NeedsBindlessChecking(const Instruction* inst) {
  uint32_t imageId;
  switch (inst->opcode()) {
    // TODO(greg-lunarg): Add all other descriptor-based references
    case SpvOp::SpvOpImageSampleImplicitLod:
    case SpvOp::SpvOpImageSampleExplicitLod:
      imageId = inst->GetSingleWordInOperand(kSpvImageSampleImageIdInIdx);
      break;
    default:
      return false;
  }
  // If VK_EXT_descriptor_indexing is defined, all descriptors need to
  // be checked if they are written.
  if (ext_descriptor_indexing_defined_)
    return true;
  Instruction* imageInst = get_def_use_mgr()->GetDef(imageId);
  if (imageInst->opcode() == SpvOp::SpvOpSampledImage) {
    imageId = imageInst->GetSingleWordInOperand(kSpvSampledImageImageIdInIdx);
    imageInst = get_def_use_mgr()->GetDef(imageId);
  }
  assert(imageInst->opcode() == SpvOp::SpvOpLoad);
  uint32_t ptrId = imageInst->GetSingleWordInOperand(kSpvLoadPtrIdInIdx);
  Instruction* ptrInst = get_def_use_mgr()->GetDef(ptrId);
  return ptrInst->opcode() == SpvOp::SpvOpAccessChain;
}

void InstBindlessCheckPass::GenDebugOutputCode() {

}

void InstBindlessCheckPass::GenBindlessCheckCode(
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
    std::vector<std::unique_ptr<Instruction>>* new_vars,
    BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr) {
  uint32_t imageId;
  switch (ref_inst_itr->opcode()) {
    // TODO(greg-lunarg): Add all other descriptor-based references
  case SpvOp::SpvOpImageSampleImplicitLod:
  case SpvOp::SpvOpImageSampleExplicitLod:
    imageId =
        ref_inst_itr->GetSingleWordInOperand(kSpvImageSampleImageIdInIdx);
    break;
  default:
    assert(false && "unexpected bindless instruction");
  }
  Instruction* imageInst = get_def_use_mgr()->GetDef(imageId);
  if (imageInst->opcode() == SpvOp::SpvOpSampledImage) {
    imageId = imageInst->GetSingleWordInOperand(kSpvSampledImageImageIdInIdx);
    imageInst = get_def_use_mgr()->GetDef(imageId);
  }
  assert(imageInst->opcode() == SpvOp::SpvOpLoad && "missing bindless load");
  uint32_t ptrId = imageInst->GetSingleWordInOperand(kSpvLoadPtrIdInIdx);
  Instruction* ptrInst = get_def_use_mgr()->GetDef(ptrId);
  if (ptrInst->opcode() == SpvOp::SpvOpAccessChain) {
    // Check index against upper bound
    assert(ptrInst->NumInOperands() == 2 &&
        "unexpected bindless index number");
    uint32_t indexId =
        ptrInst->GetSingleWordInOperand(kSpvAccessChainIndex0IdInIdx);
    Instruction* indexInst = get_def_use_mgr()->GetDef(indexId);
    ptrId = ptrInst->GetSingleWordInOperand(kSpvAccessChainBaseIdInIdx);
    ptrInst = get_def_use_mgr()->GetDef(ptrId);
    assert(ptrInst->opcode() == SpvOpVariable);
    uint32_t varTypeId = ptrInst->type_id();
    Instruction* varTypeInst = get_def_use_mgr()->GetDef(varTypeId);
    assert(varTypeInst->opcode() == SpvOpTypePointer);
    uint32_t ptrTypeId =
        varTypeInst->GetSingleWordInOperand(kSpvTypePointerTypeIdInIdx);
    Instruction* ptrTypeInst = get_def_use_mgr()->GetDef(ptrTypeId);
    // TODO(greg-lunarg): Support runtime array
    assert(ptrTypeInst->opcode() == SpvOpTypeArray);
    uint32_t lengthId =
        ptrTypeInst->GetSingleWordInOperand(kSpvTypeArrayLengthIdInIdx);
    Instruction* lengthInst = get_def_use_mgr()->GetDef(lengthId);
    if (indexInst->opcode() == SpvOpConstant &&
        lengthInst->opcode() == SpvOpConstant) {
      if (indexInst->GetSingleWordInOperand(kSpvConstantValueInIdx) >=
          lengthInst->GetSingleWordInOperand(kSpvConstantValueInIdx))
        GenDebugOutputCode();
    }
    else {

    }
  }
}

bool InstBindlessCheckPass::InstBindlessCheck(Function* func) {
  bool modified = false;
  // Using block iterators here because of block erasures and insertions.
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end();) {
      if (NeedsBindlessChecking(&*ii)) {
        // Instrument call.
        std::vector<std::unique_ptr<BasicBlock>> newBlocks;
        std::vector<std::unique_ptr<Instruction>> newVars;
        GenBindlessCheckCode(&newBlocks, &newVars, ii, bi);
        // Update succeeding phis with label of new last block.
        size_t newBlocksSize = newBlocks.size();
        assert(newBlocksSize > 1);
        UpdateSucceedingPhis(newBlocks);
        // Replace original block with new block(s).
        bi = bi.Erase();
        for (auto& bb : newBlocks) {
          bb->SetParent(func);
        }
        bi = bi.InsertBefore(&newBlocks);
        // Reset block iterator to last new block
        for (size_t i = 0; i < newBlocksSize - 1; i++) ++bi;
        // Insert new function variables.
        if (newVars.size() > 0)
          func->begin()->begin().InsertBefore(std::move(newVars));
        // Restart instrumenting at beginning of last new block.
        ii = bi->begin();
        modified = true;
      } else {
        ++ii;
      }
    }
  }
  return modified;
}

void InstBindlessCheckPass::InitializeInstBindlessCheck() {
  // Look for related extensions
  ext_descriptor_indexing_defined_ = false;
  for (auto& ei : get_module()->extensions()) {
    const char* extName =
      reinterpret_cast<const char*>(&ei.GetInOperand(0).words[0]);
    if (strcmp(extName, "SPV_EXT_descriptor_indexing") == 0) {
      ext_descriptor_indexing_defined_ = true;
      break;
    }
  }
}

Pass::Status InstBindlessCheckPass::ProcessImpl() {
  // Attempt exhaustive inlining on each entry point function in module
  ProcessFunction pfn = [this](Function* fp) { return InstBindlessCheck(fp); };
  bool modified = ProcessEntryPointCallTree(pfn, get_module());
  // TODO(greg-lunarg): If modified, do CFGCleanup
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

InstBindlessCheckPass::InstBindlessCheckPass() = default;

Pass::Status InstBindlessCheckPass::Process() {
  InitializeInstrument();
  InitializeInstBindlessCheck();
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
