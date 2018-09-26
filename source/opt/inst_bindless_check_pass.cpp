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

#include "source/opt/ir_builder.h"

// Input Operand Indices
static const int kSpvImageSampleImageIdInIdx = 0;
static const int kSpvSampledImageImageIdInIdx = 0;
static const int kSpvSampledImageSamplerIdInIdx = 1;
static const int kSpvImageSampledImageIdInIdx = 0;
static const int kSpvLoadPtrIdInIdx = 0;
static const int kSpvAccessChainBaseIdInIdx = 0;
static const int kSpvAccessChainIndex0IdInIdx = 1;
static const int kSpvTypePointerTypeIdInIdx = 1;
static const int kSpvTypeArrayLengthIdInIdx = 1;
static const int kSpvConstantValueInIdx = 0;

// Bindless-specific Output Record Offsets
static const int kInstBindlessOutError = 0;
static const int kInstBindlessOutDescIndex = 1;
static const int kInstBindlessOutDescBound = 2;
static const int kInstBindlessOutRecordSize = 3;

namespace spvtools {
namespace opt {

void InstBindlessCheckPass::GenBindlessCheckCode(
    BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr,
    uint32_t function_idx,
    uint32_t instruction_idx,
    uint32_t stage_idx,
  std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
  // Look for reference through bindless descriptor. If not, return.
  std::unique_ptr<BasicBlock> new_blk_ptr;
  uint32_t imageId;
  switch (ref_inst_itr->opcode()) {
    case SpvOp::SpvOpImageSampleImplicitLod:
    case SpvOp::SpvOpImageSampleExplicitLod:
    case SpvOp::SpvOpImageSampleDrefImplicitLod:
    case SpvOp::SpvOpImageSampleDrefExplicitLod:
    case SpvOp::SpvOpImageSampleProjImplicitLod:
    case SpvOp::SpvOpImageSampleProjExplicitLod:
    case SpvOp::SpvOpImageSampleProjDrefImplicitLod:
    case SpvOp::SpvOpImageSampleProjDrefExplicitLod:
    case SpvOp::SpvOpImageGather:
    case SpvOp::SpvOpImageDrefGather:
    case SpvOp::SpvOpImageQueryLod:
    case SpvOp::SpvOpImageSparseSampleImplicitLod:
    case SpvOp::SpvOpImageSparseSampleExplicitLod:
    case SpvOp::SpvOpImageSparseSampleDrefImplicitLod:
    case SpvOp::SpvOpImageSparseSampleDrefExplicitLod:
    case SpvOp::SpvOpImageSparseSampleProjImplicitLod:
    case SpvOp::SpvOpImageSparseSampleProjExplicitLod:
    case SpvOp::SpvOpImageSparseSampleProjDrefImplicitLod:
    case SpvOp::SpvOpImageSparseSampleProjDrefExplicitLod:
    case SpvOp::SpvOpImageSparseGather:
    case SpvOp::SpvOpImageSparseDrefGather:
    case SpvOp::SpvOpImageFetch:
    case SpvOp::SpvOpImageRead:
    case SpvOp::SpvOpImageQueryFormat:
    case SpvOp::SpvOpImageQueryOrder:
    case SpvOp::SpvOpImageQuerySizeLod:
    case SpvOp::SpvOpImageQuerySize:
    case SpvOp::SpvOpImageQueryLevels:
    case SpvOp::SpvOpImageQuerySamples:
    case SpvOp::SpvOpImageSparseFetch:
    case SpvOp::SpvOpImageSparseRead:
    case SpvOp::SpvOpImageWrite:
      imageId =
          ref_inst_itr->GetSingleWordInOperand(kSpvImageSampleImageIdInIdx);
      break;
    default:
      return;
  }
  Instruction* imageInst = get_def_use_mgr()->GetDef(imageId);
  uint32_t loadId;
  Instruction* loadInst;
  if (imageInst->opcode() == SpvOp::SpvOpSampledImage) {
    loadId = imageInst->GetSingleWordInOperand(kSpvSampledImageImageIdInIdx);
    loadInst = get_def_use_mgr()->GetDef(loadId);
  }
  else if (imageInst->opcode() == SpvOp::SpvOpImage) {
    loadId = imageInst->GetSingleWordInOperand(kSpvImageSampledImageIdInIdx);
    loadInst = get_def_use_mgr()->GetDef(loadId);
  }
  else {
    loadId = imageId;
    loadInst = imageInst;
    imageId = 0;
  }
  if (loadInst->opcode() != SpvOp::SpvOpLoad) {
    // TODO(greg-lunarg): Handle additional possibilities
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
  if (ptrInst->opcode() != SpvOpVariable) {
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
  uint32_t errorId = GetUintConstantId(kInstErrorBindlessBounds);
  uint32_t lengthId =
    ptrTypeInst->GetSingleWordInOperand(kSpvTypeArrayLengthIdInIdx);
  // Generate full runtime bounds test code with true branch
  // being full reference and false branch being debug output and zero
  // for the referenced value.
  MovePreludeCode(ref_inst_itr, ref_block_itr, &new_blk_ptr);
  InstructionBuilder builder(context(), &*new_blk_ptr,
      IRContext::kAnalysisDefUse);
  Instruction* ult_inst = builder.AddBinaryOp(GetBoolId(), SpvOpULessThan,
      indexId, lengthId);
  uint32_t mergeBlkId = TakeNextId();
  uint32_t validBlkId = TakeNextId();
  uint32_t invalidBlkId = TakeNextId();
  std::unique_ptr<Instruction> mergeLabel(NewLabel(mergeBlkId));
  std::unique_ptr<Instruction> validLabel(NewLabel(validBlkId));
  std::unique_ptr<Instruction> invalidLabel(NewLabel(invalidBlkId));
  (void) builder.AddConditionalBranch(ult_inst->result_id(), validBlkId,
      invalidBlkId, mergeBlkId, SpvSelectionControlMaskNone);
  // Close selection block and gen valid reference block
  new_blocks->push_back(std::move(new_blk_ptr));
  new_blk_ptr.reset(new BasicBlock(std::move(validLabel)));
  builder.SetInsertPoint(&*new_blk_ptr);
  // Clone descriptor load
  Instruction* newLoadInst = builder.AddLoad(loadInst->type_id(),
      loadInst->GetSingleWordInOperand(kSpvLoadPtrIdInIdx));
  uint32_t newLoadId = newLoadInst->result_id();
  get_decoration_mgr()->CloneDecorations(loadInst->result_id(), newLoadId);
  uint32_t newImageId = newLoadId;
  // Clone Image/SampledImage with new load, if needed
  if (imageId != 0) {
    if (imageInst->opcode() == SpvOp::SpvOpSampledImage) {
      Instruction* newImageInst = builder.AddBinaryOp(imageInst->type_id(),
          SpvOpSampledImage, newLoadId, imageInst->GetSingleWordInOperand(
              kSpvSampledImageSamplerIdInIdx));
      newImageId = newImageInst->result_id();
    }
    else {
      assert(imageInst->opcode() == SpvOp::SpvOpImage && "expecting OpImage");
      Instruction* newImageInst = builder.AddUnaryOp(imageInst->type_id(),
          SpvOpImage, newLoadId);
      newImageId = newImageInst->result_id();
    }
    get_decoration_mgr()->CloneDecorations(imageId, newImageId);
  }
  // Clone original reference using new image code
  std::unique_ptr<Instruction> newRefInst(ref_inst_itr->Clone(context()));
  uint32_t refResultId = ref_inst_itr->result_id();
  uint32_t newRefId = 0;
  if (refResultId != 0) {
    newRefId = TakeNextId();
    newRefInst->SetResultId(newRefId);
  }
  newRefInst->SetInOperand(kSpvImageSampleImageIdInIdx, { newImageId });
  // Register new reference and add to new block
  get_def_use_mgr()->AnalyzeInstDefUse(&*newRefInst);
  new_blk_ptr->AddInstruction(std::move(newRefInst));
  if (newRefId != 0)
    get_decoration_mgr()->CloneDecorations(refResultId, newRefId);
  // Close valid block and gen invalid block
  (void) builder.AddBranch(mergeBlkId);
  new_blocks->push_back(std::move(new_blk_ptr));
  new_blk_ptr.reset(new BasicBlock(std::move(invalidLabel)));
  builder.SetInsertPoint(&*new_blk_ptr);
  GenDebugOutputCode(function_idx, instruction_idx,
      stage_idx, { errorId, indexId, lengthId }, new_blocks,
      &new_blk_ptr);
  builder.SetInsertPoint(&*new_blk_ptr);
  // Remember last invalid block id
  uint32_t lastInvalidBlkId = new_blk_ptr->GetLabelInst()->result_id();
  // Gen zero for invalid  reference
  uint32_t refTypeId = ref_inst_itr->type_id();
  uint32_t nullId = 0;
  if (newRefId != 0)
    nullId = GetNullId(refTypeId);
  // Close invalid block and gen merge block
  (void) builder.AddBranch(mergeBlkId);
  new_blocks->push_back(std::move(new_blk_ptr));
  new_blk_ptr.reset(new BasicBlock(std::move(mergeLabel)));
  builder.SetInsertPoint(&*new_blk_ptr);
  // Gen phi of new reference and zero, if necessary. Use result id of
  // original reference so we don't have to do a replace. Kill original
  // reference before reusing its id.
  context()->KillInst(&*ref_inst_itr);
  if (newRefId != 0)
    (void) builder.AddPhi(refTypeId,
        { newRefId, validBlkId, nullId, lastInvalidBlkId }, refResultId);
  MovePostludeCode(ref_block_itr, &new_blk_ptr);
  // Add remainder/merge block to new blocks
  new_blocks->push_back(std::move(new_blk_ptr));
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
  // Perform instrumentation on each entry point function in module
  InstProcessFunction pfn = [this](
      BasicBlock::iterator ref_inst_itr,
      UptrVectorIterator<BasicBlock> ref_block_itr,
      uint32_t function_idx,
      uint32_t instruction_idx,
      uint32_t stage_idx,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks) {
    return GenBindlessCheckCode(ref_inst_itr, ref_block_itr,
        function_idx, instruction_idx, stage_idx, new_blocks); };
  bool modified = InstProcessEntryPointCallTree(pfn, get_module());
  // This pass does not update inst->blk info
  context()->InvalidateAnalyses(IRContext::kAnalysisInstrToBlockMapping);
  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

Pass::Status InstBindlessCheckPass::Process() {
  InitializeInstBindlessCheck();
  return ProcessImpl();
}

}  // namespace opt
}  // namespace spvtools
