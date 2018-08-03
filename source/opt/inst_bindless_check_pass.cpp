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

void InstBindlessCheckPass::GenDebugOutputCode() {
}

void InstBindlessCheckPass::GenBindlessCheckCode(
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
    std::vector<std::unique_ptr<Instruction>>* new_vars,
    BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr) {
  std::unique_ptr<BasicBlock> new_blk_ptr;
  uint32_t sampledImageId;
  switch (ref_inst_itr->opcode()) {
    // TODO(greg-lunarg): Add all other descriptor-based references
    case SpvOp::SpvOpImageSampleImplicitLod:
    case SpvOp::SpvOpImageSampleExplicitLod:
      sampledImageId =
          ref_inst_itr->GetSingleWordInOperand(kSpvImageSampleImageIdInIdx);
      break;
    default:
      return;
  }
  Instruction* sampledImageInst = get_def_use_mgr()->GetDef(sampledImageId);
  uint32_t imageId = 0;
  Instruction* imageInst;
  if (sampledImageInst->opcode() == SpvOp::SpvOpSampledImage) {
    imageId = imageInst->GetSingleWordInOperand(kSpvSampledImageImageIdInIdx);
    imageInst = get_def_use_mgr()->GetDef(imageId);
  }
  else {
    imageId = sampledImageId;
    imageInst = sampledImageInst;
    sampledImageId = 0;
  }
  if (imageInst->opcode() != SpvOp::SpvOpLoad) {
    assert(false && "unexpected image value");
    return;
  }
  uint32_t ptrId = imageInst->GetSingleWordInOperand(kSpvLoadPtrIdInIdx);
  Instruction* ptrInst = get_def_use_mgr()->GetDef(ptrId);
  // TODO(greg-lunarg): Check non-indexed descriptor refs
  if (ptrInst->opcode() != SpvOp::SpvOpAccessChain)
    return;
  // Check index against upper bound
  if (ptrInst->NumInOperands() != 2) {
    assert(false && "unexpected bindless index number");
    return;
  }
  uint32_t indexId =
      ptrInst->GetSingleWordInOperand(kSpvAccessChainIndex0IdInIdx);
  Instruction* indexInst = get_def_use_mgr()->GetDef(indexId);
  ptrId = ptrInst->GetSingleWordInOperand(kSpvAccessChainBaseIdInIdx);
  ptrInst = get_def_use_mgr()->GetDef(ptrId);
  if (ptrInst->opcode() == SpvOpVariable) {
    assert(false && "unexpected bindless base");
    return;
  }
  uint32_t varTypeId = ptrInst->type_id();
  Instruction* varTypeInst = get_def_use_mgr()->GetDef(varTypeId);
  uint32_t ptrTypeId =
      varTypeInst->GetSingleWordInOperand(kSpvTypePointerTypeIdInIdx);
  Instruction* ptrTypeInst = get_def_use_mgr()->GetDef(ptrTypeId);
  // TODO(greg-lunarg): Support runtime array
  if (ptrTypeInst->opcode() != SpvOpTypeArray)
    return;
  uint32_t lengthId =
      ptrTypeInst->GetSingleWordInOperand(kSpvTypeArrayLengthIdInIdx);
  Instruction* lengthInst = get_def_use_mgr()->GetDef(lengthId);
  // If index and bound both compile-time constants and index >= bound,
  // just generate debug output code. Otherwise generate bounds test
  // code. True branch is full reference; false branch is debug output.
  if (indexInst->opcode() == SpvOpConstant &&
      lengthInst->opcode() == SpvOpConstant) {
    if (indexInst->GetSingleWordInOperand(kSpvConstantValueInIdx) >=
        lengthInst->GetSingleWordInOperand(kSpvConstantValueInIdx)) {
      MovePreludeCode(ref_inst_itr, ref_block_itr, &new_blk_ptr);
      GenDebugOutputCode();
    }
  }
  else {
    MovePreludeCode(ref_inst_itr, ref_block_itr, &new_blk_ptr);
    analysis::Bool boolTy;
    uint32_t boolTyId = context()->get_type_mgr()->GetTypeInstruction(&boolTy);
    uint32_t ultId = AddBinaryOp(boolTyId, SpvOpULessThan, indexId, lengthId,
        &new_blk_ptr);
    uint32_t mergeBlkId = TakeNextId();
    uint32_t validBlkId = TakeNextId();
    uint32_t invalidBlkId = TakeNextId();
    AddSelectionMerge(mergeBlkId, SpvSelectionControlMaskNone, &new_blk_ptr);
    AddBranchCond(ultId, validBlkId, invalidBlkId, &new_blk_ptr);
    // Gen valid code block
    new_blocks->push_back(std::move(new_blk_ptr));
    new_blk_ptr.reset(new BasicBlock(NewLabel(validBlkId)));
    // Clone descriptor load
    std::unique_ptr<Instruction> newLoadInst(imageInst->Clone(context()));
    uint32_t newLoadId = TakeNextId();
    newLoadInst->SetResultId(newLoadId);
  }
}

bool InstBindlessCheckPass::InstBindlessCheck(Function* func) {
  bool modified = false;
  std::vector<std::unique_ptr<BasicBlock>> newBlocks;
  std::vector<std::unique_ptr<Instruction>> newVars;
  // Using block iterators here because of block erasures and insertions.
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    preCallSB_.clear();
    postCallSB_.clear();
    for (auto ii = bi->begin(); ii != bi->end();) {
      GenBindlessCheckCode(&newBlocks, &newVars, ii, bi);
      if (newBlocks.size() > 0) {
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
        modified = true;
        // Restart instrumenting at beginning of last new block.
        ii = bi->begin();
        assert(newBlocks.size() == 0);
        assert(newVars.size() == 0);
      } else {
        ++ii;
      }
    }
  }
  return modified;
}

void InstBindlessCheckPass::InitializeInstBindlessCheck() {
  // Initialize base class
  InitializeInstrument();
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
  // This pass does not update def/use info
  context()->InvalidateAnalyses(IRContext::kAnalysisDefUse);
  // TODO(greg-lunarg): If modified, do CFGCleanup, DCE
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

InstBindlessCheckPass::InstBindlessCheckPass() = default;

Pass::Status InstBindlessCheckPass::Process() {
  InitializeInstBindlessCheck();
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
