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

// Operand indices
static const int kSpvImageSampleImageIdIdx = 2;

// Input Operand Indices
static const int kSpvImageSampleImageIdInIdx = 0;
static const int kSpvSampledImageImageIdInIdx = 0;
static const int kSpvSampledImageSamplerIdInIdx = 1;
static const int kSpvLoadPtrIdInIdx = 0;
static const int kSpvAccessChainBaseIdInIdx = 0;
static const int kSpvAccessChainIndex0IdInIdx = 0;
static const int kSpvTypePointerTypeIdInIdx = 0;
static const int kSpvTypeArrayLengthIdInIdx = 1;
static const int kSpvConstantValueInIdx = 0;

namespace spvtools {
namespace opt {

void InstBindlessCheckPass::GenDebugOutputCode(
    std::unique_ptr<BasicBlock>* new_blk_ptr) {
}

void InstBindlessCheckPass::GenBindlessCheckCode(
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
    std::vector<std::unique_ptr<Instruction>>* new_vars,
    BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr) {
  // Look for reference through bindless descriptor. If not, return.
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
  uint32_t loadId;
  Instruction* loadInst;
  if (sampledImageInst->opcode() == SpvOp::SpvOpSampledImage) {
    loadId = sampledImageInst->GetSingleWordInOperand(kSpvSampledImageImageIdInIdx);
    loadInst = get_def_use_mgr()->GetDef(loadId);
  }
  else {
    loadId = sampledImageId;
    loadInst = sampledImageInst;
    sampledImageId = 0;
  }
  if (loadInst->opcode() != SpvOp::SpvOpLoad) {
    assert(false && "unexpected image value");
    return;
  }
  uint32_t ptrId = loadInst->GetSingleWordInOperand(kSpvLoadPtrIdInIdx);
  Instruction* ptrInst = get_def_use_mgr()->GetDef(ptrId);
  // Check descriptor index against upper bound
  // TODO(greg-lunarg): Check descriptor to make sure it is written.
  if (ptrInst->opcode() != SpvOp::SpvOpAccessChain)
    return;
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
  // TODO(greg-lunarg): Check descriptor index against runtime array
  // size.
  if (ptrTypeInst->opcode() != SpvOpTypeArray)
    return;
  // If index and bound both compile-time constants and index >= bound,
  // generate debug output error code and use zero as referenced value.
  uint32_t lengthId =
    ptrTypeInst->GetSingleWordInOperand(kSpvTypeArrayLengthIdInIdx);
  Instruction* lengthInst = get_def_use_mgr()->GetDef(lengthId);
  if (indexInst->opcode() == SpvOpConstant &&
      lengthInst->opcode() == SpvOpConstant) {
    if (indexInst->GetSingleWordInOperand(kSpvConstantValueInIdx) >=
        lengthInst->GetSingleWordInOperand(kSpvConstantValueInIdx)) {
      MovePreludeCode(ref_inst_itr, ref_block_itr, &new_blk_ptr);
      GenDebugOutputCode(&new_blk_ptr);
      // Set the original reference id to zero. Kill original reference
      // before reusing id.
      uint32_t ref_type_id = ref_inst_itr->type_id();
      uint32_t ref_result_id = ref_inst_itr->result_id();
      uint32_t nullId = GetNullId(ref_type_id);
      context()->KillInst(&*ref_inst_itr);
      AddUnaryOp(ref_type_id, ref_result_id, SpvOpCopyObject,
          nullId, &new_blk_ptr);
      // Close error block and create and populate remainder block
      uint32_t remBlkId = TakeNextId();
      std::unique_ptr<Instruction> remLabel(NewLabel(remBlkId));
      AddBranch(remBlkId, &new_blk_ptr);
      new_blocks->push_back(std::move(new_blk_ptr));
      new_blk_ptr.reset(new BasicBlock(std::move(remLabel)));
      MovePostludeCode(ref_block_itr, &new_blk_ptr);
    }
  }
  // Otherwise generate full runtime bounds test code with true branch
  // being full reference and false branch being debug output and zero
  // for the referenced value.
  else {
    MovePreludeCode(ref_inst_itr, ref_block_itr, &new_blk_ptr);
    analysis::Bool boolTy;
    uint32_t boolTyId = context()->get_type_mgr()->GetTypeInstruction(&boolTy);
    uint32_t ultId = TakeNextId();
    AddBinaryOp(boolTyId, ultId, SpvOpULessThan, indexId, lengthId, &new_blk_ptr);
    uint32_t mergeBlkId = TakeNextId();
    uint32_t validBlkId = TakeNextId();
    uint32_t invalidBlkId = TakeNextId();
    std::unique_ptr<Instruction> mergeLabel(NewLabel(mergeBlkId));
    std::unique_ptr<Instruction> validLabel(NewLabel(validBlkId));
    std::unique_ptr<Instruction> invalidLabel(NewLabel(invalidBlkId));
    AddSelectionMerge(mergeBlkId, SpvSelectionControlMaskNone, &new_blk_ptr);
    AddBranchCond(ultId, validBlkId, invalidBlkId, &new_blk_ptr);
    // Close selection block and gen valid reference block
    new_blocks->push_back(std::move(new_blk_ptr));
    new_blk_ptr.reset(new BasicBlock(std::move(validLabel)));
    // Clone descriptor load
    std::unique_ptr<Instruction> newLoadInst(loadInst->Clone(context()));
    uint32_t newLoadId = TakeNextId();
    newLoadInst->SetResultId(newLoadId);
    get_def_use_mgr()->AnalyzeInstDefUse(&*newLoadInst);
    new_blk_ptr->AddInstruction(std::move(newLoadInst));
    get_decoration_mgr()->CloneDecorations(loadInst->result_id(), newLoadId);
    uint32_t imageId = newLoadId;
    // Clone SampledImage with new load, if needed
    if (sampledImageId != 0) {
      imageId = TakeNextId();
      AddBinaryOp(sampledImageInst->type_id(), imageId, SpvOpSampledImage,
          newLoadId, sampledImageInst->GetSingleWordInOperand(
          kSpvSampledImageSamplerIdInIdx), &new_blk_ptr);
      get_decoration_mgr()->CloneDecorations(sampledImageInst->result_id(),
          imageId);
    }
    // Clone original reference using new image code
    std::unique_ptr<Instruction> newRefInst(ref_inst_itr->Clone(context()));
    uint32_t newRefId = TakeNextId();
    newRefInst->SetResultId(newRefId);
    switch (ref_inst_itr->opcode()) {
      // TODO(greg-lunarg): Add all other descriptor-based references
      case SpvOp::SpvOpImageSampleImplicitLod:
      case SpvOp::SpvOpImageSampleExplicitLod:
        newRefInst->SetOperand(kSpvImageSampleImageIdIdx, {imageId});
        break;
      default:
        assert(false && "unexpected reference opcode");
        break;
    }
    // Register new reference
    get_def_use_mgr()->AnalyzeInstDefUse(&*newRefInst);
    get_decoration_mgr()->CloneDecorations(ref_inst_itr->result_id(),
        newRefId);
    // Close valid block and gen invalid block
    AddBranch(mergeBlkId, &new_blk_ptr);
    new_blocks->push_back(std::move(new_blk_ptr));
    new_blk_ptr.reset(new BasicBlock(std::move(invalidLabel)));
    GenDebugOutputCode(&new_blk_ptr);
    // Gen zero for invalid  reference
    uint32_t ref_type_id = ref_inst_itr->type_id();
    uint32_t nullId = GetNullId(ref_type_id);
    // Close invalid block and gen merge block
    AddBranch(mergeBlkId, &new_blk_ptr);
    new_blocks->push_back(std::move(new_blk_ptr));
    new_blk_ptr.reset(new BasicBlock(std::move(mergeLabel)));
    // Gen phi of new reference and zero. Use result id of original reference
    // so we don't have to do a replace. Kill original reference before reusing
    // its id.
    context()->KillInst(&*ref_inst_itr);
    AddPhi(ref_type_id, ref_inst_itr->result_id(), newRefId, validBlkId,
        nullId, invalidBlkId, &new_blk_ptr);
    // Move remainder of original block instructions into merge block
    MovePostludeCode(ref_block_itr, &new_blk_ptr);
  }
  // Push remainder/merge block
  new_blocks->push_back(std::move(new_blk_ptr));
}

bool InstBindlessCheckPass::InstBindlessCheck(Function* func) {
  bool modified = false;
  std::vector<std::unique_ptr<BasicBlock>> newBlocks;
  std::vector<std::unique_ptr<Instruction>> newVars;
  // Using block iterators here because of block erasures and insertions.
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    for (auto ii = bi->begin(); ii != bi->end();) {
      GenBindlessCheckCode(&newBlocks, &newVars, ii, bi);
      if (newBlocks.size() == 0) {
        ++ii;
        continue;
      }
      // If there are new blocks we know there will always be two or
      // more, so update succeeding phis with label of new last block.
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
