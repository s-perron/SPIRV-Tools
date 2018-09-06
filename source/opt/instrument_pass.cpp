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

#include "instrument_pass.h"

#include "source/cfa.h"

// Debug Buffer Offsets
static const int kDebugOutputSizeOffset = 0;
static const int kDebugOutputDataOffset = 1;

// Common Output Record Offsets
static const int kInstCommonOutSize = 0;
static const int kInstCommonOutShaderId = 1;
static const int kInstCommonOutFunctionIdx = 2;
static const int kInstCommonOutInstructionIdx = 3;
static const int kInstCommonOutStageIdx = 4;

// Frag Shader Output Record Offsets
static const int kInstFragOutFragCoordX = 5;
static const int kInstFragOutFragCoordY = 6;
static const int kInstFragOutRecordSize = 7;

// Indices of operands in SPIR-V instructions
static const int kSpvFunctionCallFunctionId = 2;
static const int kSpvFunctionCallArgumentId = 3;
static const int kSpvReturnValueId = 0;
static const int kSpvLoopMergeMergeBlockId = 0;
static const int kSpvLoopMergeContinueTargetIdInIdx = 1;
static const int kEntryPointExecutionModelInIdx = 0;
static const int kEntryPointFunctionIdInIdx = 1;
static const int kEntryPointInterfaceInIdx = 3;
static const int kSpvDecorateTargetIdInIdx = 0;
static const int kSpvDecorateDecorationInIdx = 1;
static const int kSpvDecorateBuiltinInIdx = 2;
static const int kSpvMemberDecorateDecorationInIdx = 2;
static const int kSpvMemberDecorateBuiltinInIdx = 3;

namespace spvtools {
namespace opt {

void InstrumentPass::MovePreludeCode(
    BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr,
    std::unique_ptr<BasicBlock>* new_blk_ptr) {
  preCallSB_.clear();
  postCallSB_.clear();
  // Reuse label from ref block. Kill previous label
  // before reusing.
  uint32_t ref_blk_id = ref_block_itr->id();
  context()->KillInst(ref_block_itr->GetLabelInst());
  new_blk_ptr->reset(new BasicBlock(NewLabel(ref_blk_id)));
  // Move contents of original ref block up to ref instruction.
  for (auto cii = ref_block_itr->begin(); cii != ref_inst_itr;
      cii = ref_block_itr->begin()) {
    Instruction* inst = &*cii;
    inst->RemoveFromList();
    std::unique_ptr<Instruction> mv_ptr(inst);
    // Remember same-block ops for possible regeneration.
    if (IsSameBlockOp(&*mv_ptr)) {
      auto* sb_inst_ptr = mv_ptr.get();
      preCallSB_[mv_ptr->result_id()] = sb_inst_ptr;
    }
    (*new_blk_ptr)->AddInstruction(std::move(mv_ptr));
  }
}

void InstrumentPass::MovePostludeCode(
  UptrVectorIterator<BasicBlock> ref_block_itr,
  std::unique_ptr<BasicBlock>* new_blk_ptr) {
  // new_blk_ptr->reset(new BasicBlock(NewLabel(ref_block_itr->id())));
  // Move contents of original ref block.
  for (auto cii = ref_block_itr->begin(); cii != ref_block_itr->end();
      cii = ref_block_itr->begin()) {
    Instruction* inst = &*cii;
    inst->RemoveFromList();
    std::unique_ptr<Instruction> mv_inst(inst);
    // Regenerate any same-block instruction that has not been seen in the
    // current block.
    if (preCallSB_.size() > 0) {
      CloneSameBlockOps(&mv_inst, &postCallSB_, &preCallSB_, new_blk_ptr);
      // Remember same-block ops in this block.
      if (IsSameBlockOp(&*mv_inst)) {
        const uint32_t rid = mv_inst->result_id();
        postCallSB_[rid] = rid;
      }
    }
    (*new_blk_ptr)->AddInstruction(std::move(mv_inst));
  }
}

void InstrumentPass::AddUnaryOp(uint32_t type_id, uint32_t result_id,
                                     SpvOp opcode, uint32_t operand,
                                     std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newUnOp(
      new Instruction(context(), opcode, type_id, result_id,
      { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newUnOp);
  (*block_ptr)->AddInstruction(std::move(newUnOp));
}

void InstrumentPass::AddBinaryOp(uint32_t type_id, uint32_t result_id,
  SpvOp opcode, uint32_t operand1, uint32_t operand2,
  std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newBinOp(
    new Instruction(context(), opcode, type_id, result_id,
    { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand1 } },
    { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand2 } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newBinOp);
  (*block_ptr)->AddInstruction(std::move(newBinOp));
}

void InstrumentPass::AddTernaryOp(uint32_t type_id, uint32_t result_id,
  SpvOp opcode, uint32_t operand1, uint32_t operand2, uint32_t operand3,
  std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newTernOp(
    new Instruction(context(), opcode, type_id, result_id,
    { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand1 } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand2 } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand3 } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newTernOp);
  (*block_ptr)->AddInstruction(std::move(newTernOp));
}

void InstrumentPass::AddQuadOp(uint32_t type_id, uint32_t result_id,
  SpvOp opcode, uint32_t operand1, uint32_t operand2, uint32_t operand3,
  uint32_t operand4, std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newQuadOp(
    new Instruction(context(), opcode, type_id, result_id,
    { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand1 } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand2 } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand3 } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand4 } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newQuadOp);
  (*block_ptr)->AddInstruction(std::move(newQuadOp));
}

void InstrumentPass::AddExtractOp(uint32_t type_id, uint32_t result_id,
  uint32_t operand1, uint32_t operand2,
  std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newBinOp(
    new Instruction(context(), SpvOpCompositeExtract, type_id, result_id,
    { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ operand1 } },
    { spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,{ operand2 } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newBinOp);
  (*block_ptr)->AddInstruction(std::move(newBinOp));
}

void InstrumentPass::AddArrayLength(uint32_t result_id,
    uint32_t struct_ptr_id, uint32_t member_idx,
    std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newALenOp(
      new Instruction(context(), SpvOpArrayLength,
          GetUintId(), result_id,
          { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ struct_ptr_id } },
            { spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {
              member_idx } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newALenOp);
  (*block_ptr)->AddInstruction(std::move(newALenOp));
}

uint32_t InstrumentPass::FindBuiltin(uint32_t builtin_val) {
  for (auto& a : get_module()->annotations()) {
    if (a.opcode() == SpvOpDecorate) {
      if (a.GetSingleWordInOperand(kSpvDecorateDecorationInIdx) !=
          SpvDecorationBuiltIn)
        continue;
      if (a.GetSingleWordInOperand(kSpvDecorateBuiltinInIdx) != builtin_val)
        continue;
      uint32_t target_id = a.GetSingleWordInOperand(kSpvDecorateTargetIdInIdx);
      // TODO(greg-lunarg): Support group builtins
      Instruction* b_var = context()->get_def_use_mgr()->GetDef(target_id);
      if (b_var->opcode() != SpvOpVariable)
        assert(false && "unexpected group builtin");
      return target_id;
    }
    else if (a.opcode() == SpvOpMemberDecorate) {
      if (a.GetSingleWordInOperand(kSpvMemberDecorateDecorationInIdx) !=
          SpvDecorationBuiltIn)
        continue;
      if (a.GetSingleWordInOperand(kSpvMemberDecorateBuiltinInIdx) !=
          builtin_val)
        continue;
      // TODO(greg-lunarg): Support member builtins
      assert(false && "unexpected member builtin");
      return 0;
    }
  }
  return 0;
}

void InstrumentPass::AddDecoration(uint32_t inst_id, uint32_t decoration,
    uint32_t decoration_value) {
  std::unique_ptr<Instruction> newDecoOp(
    new Instruction(context(), SpvOpDecorate, 0, 0,
      { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,
          { inst_id } },
        { spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
          { decoration } },
        { spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
          { decoration_value } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newDecoOp);
  get_decoration_mgr()->AddDecoration(&*newDecoOp);
  get_module()->AddAnnotationInst(std::move(newDecoOp));
}

void InstrumentPass::AddSelectionMerge(
  uint32_t mergeBlockId, uint32_t selControl,
  std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newSMOp(
    new Instruction(context(), SpvOpSelectionMerge, 0, 0,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {mergeBlockId}},
       {spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER, {selControl}}}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newSMOp);
  (*block_ptr)->AddInstruction(std::move(newSMOp));
}

void InstrumentPass::AddBranch(uint32_t label_id,
                           std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newBranch(
      new Instruction(context(), SpvOpBranch, 0, 0,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {label_id}}}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newBranch);
  (*block_ptr)->AddInstruction(std::move(newBranch));
}

void InstrumentPass::AddBranchCond(uint32_t cond_id, uint32_t true_id,
                               uint32_t false_id,
                               std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newBranch(
      new Instruction(context(), SpvOpBranchConditional, 0, 0,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_ID, {cond_id}},
                       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {true_id}},
                       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {false_id}}}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newBranch);
  (*block_ptr)->AddInstruction(std::move(newBranch));
}

void InstrumentPass::AddPhi(uint32_t type_id, uint32_t result_id, uint32_t var0_id,
  uint32_t parent0_id, uint32_t var1_id, uint32_t parent1_id,
  std::unique_ptr<BasicBlock>* block_ptr) {
  std::unique_ptr<Instruction> newPhi(
    new Instruction(context(), SpvOpPhi, type_id, result_id,
    { { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ var0_id } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ parent0_id } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ var1_id } },
      { spv_operand_type_t::SPV_OPERAND_TYPE_ID,{ parent1_id } } }));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newPhi);
  (*block_ptr)->AddInstruction(std::move(newPhi));
}

std::unique_ptr<Instruction> InstrumentPass::NewLabel(uint32_t label_id) {
  std::unique_ptr<Instruction> newLabel(
      new Instruction(context(), SpvOpLabel, 0, label_id, {}));
  get_def_use_mgr()->AnalyzeInstDefUse(&*newLabel);
  return newLabel;
}

uint32_t InstrumentPass::GetNullId(uint32_t type_id) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
  const analysis::Type* type = type_mgr->GetType(type_id);
  const analysis::Constant* null_const = const_mgr->GetConstant(type, {});
  Instruction* null_inst =
      const_mgr->GetDefiningInstruction(null_const, type_id);
  return null_inst->result_id();
}

uint32_t InstrumentPass::GetUintConstantId(uint32_t u) {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::ConstantManager* const_mgr = context()->get_constant_mgr();
  analysis::Integer uint_ty(32, false);
  analysis::Type* reg_uint_ty = type_mgr->GetRegisteredType(&uint_ty);
  const analysis::Constant* uint_const = const_mgr->GetConstant(reg_uint_ty, {u});
  Instruction* uint_inst = const_mgr->GetDefiningInstruction(uint_const);
  return uint_inst->result_id();
}

void InstrumentPass::GenDebugOutputFieldCode(
    uint32_t base_offset_id,
    uint32_t field_offset,
    uint32_t field_value_id,
    std::unique_ptr<BasicBlock>* new_blk_ptr) {
  uint32_t data_idx_id = TakeNextId();
  AddBinaryOp(GetUintId(), data_idx_id, SpvOpIAdd,
      base_offset_id, GetUintConstantId(field_offset), new_blk_ptr);
  uint32_t achain_id = TakeNextId();
  AddTernaryOp(GetOutputBufferUintPtrId(), achain_id, SpvOpAccessChain,
      GetOutputBufferId(), GetUintConstantId(kDebugOutputDataOffset),
      data_idx_id, new_blk_ptr);
  AddBinaryOp(0, 0, SpvOpStore, achain_id, field_value_id, new_blk_ptr);
}

void InstrumentPass::GenCommonDebugOutputCode(
    uint32_t record_sz,
    uint32_t func_idx,
    uint32_t instruction_idx,
    uint32_t stage_idx,
    uint32_t base_offset_id,
    std::unique_ptr<BasicBlock>* new_blk_ptr) {
  // Store record size
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutSize,
      GetUintConstantId(record_sz), new_blk_ptr);
  // Store Shader Id
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutShaderId,
      GetUintConstantId(shader_id_), new_blk_ptr);
  // Store Function Idx
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutFunctionIdx,
      GetUintConstantId(func_idx), new_blk_ptr);
  // Store Instruction Idx
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutInstructionIdx,
      GetUintConstantId(instruction_idx), new_blk_ptr);
  // Store Stage Idx
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutStageIdx,
      GetUintConstantId(stage_idx), new_blk_ptr);
}

void InstrumentPass::GenFragCoordEltDebugOutputCode(
    uint32_t base_offset_id,
    uint32_t uint_frag_coord_id,
    uint32_t element,
    std::unique_ptr<BasicBlock>* new_blk_ptr) {
  uint32_t element_val_id = TakeNextId();
  AddExtractOp(GetUintId(), element_val_id,
      uint_frag_coord_id, element, new_blk_ptr);
  GenDebugOutputFieldCode(base_offset_id, kInstFragOutFragCoordX + element,
      element_val_id, new_blk_ptr);
}

void InstrumentPass::GenFragDebugOutputCode(
    uint32_t base_offset_id,
    std::unique_ptr<BasicBlock>* new_blk_ptr) {
  // Load FragCoord and convert to Uint
  uint32_t frag_coord_id = TakeNextId();
  AddUnaryOp(GetVec4FloatId(), frag_coord_id, SpvOpLoad, GetFragCoordId(),
      new_blk_ptr);
  uint32_t uint_frag_coord_id = TakeNextId();
  AddUnaryOp(GetVec4UintId(), uint_frag_coord_id, SpvOpBitcast, frag_coord_id,
      new_blk_ptr);
  for (uint32_t u = 0; u < 2u; ++u)
    GenFragCoordEltDebugOutputCode(base_offset_id,  uint_frag_coord_id, u,
        new_blk_ptr);
}

uint32_t InstrumentPass::GetStageOutputRecordSize(uint32_t stage_idx) {
  // TODO(greg-lunarg): Add support for all stages
  // TODO(greg-lunarg): Assert fragment shader
  if (stage_idx != SpvExecutionModelFragment)
    assert(false && "unexpected stage");
  return kInstFragOutRecordSize;
}

void InstrumentPass::GenDebugOutputCode(
    uint32_t func_idx,
    uint32_t instruction_idx,
    uint32_t stage_idx,
    const std::vector<uint32_t> &validation_data,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
    std::unique_ptr<BasicBlock>* new_blk_ptr) {
  // Gen test if debug output buffer size will not be exceeded.
  uint32_t obuf_record_sz = GetStageOutputRecordSize(stage_idx) +
      static_cast<uint32_t>(validation_data.size());
  uint32_t obuf_curr_sz_ac_id = TakeNextId();
  AddBinaryOp(GetOutputBufferUintPtrId(), obuf_curr_sz_ac_id, SpvOpAccessChain,
      GetOutputBufferId(), GetUintConstantId(kDebugOutputSizeOffset),
      new_blk_ptr);
  // Fetch the current debug buffer written size atomically, adding the
  // size of the record to be written.
  uint32_t obuf_curr_sz_id = TakeNextId();
  AddQuadOp(GetUintId(), obuf_curr_sz_id,
      SpvOpAtomicIAdd,
      obuf_curr_sz_ac_id,
      GetUintConstantId(SpvScopeInvocation),
      GetUintConstantId(SpvMemoryAccessMaskNone),
      GetUintConstantId(obuf_record_sz),
      new_blk_ptr);
  // Compute new written size
  uint32_t obuf_new_sz_id = TakeNextId();
  AddBinaryOp(GetUintId(), obuf_new_sz_id,
      SpvOpIAdd,
      obuf_curr_sz_id, GetUintConstantId(obuf_record_sz),
      new_blk_ptr);
  // Fetch the data bound
  uint32_t obuf_bnd_id = TakeNextId();
  AddArrayLength(obuf_bnd_id,
      GetOutputBufferId(),
      kDebugOutputDataOffset,
      new_blk_ptr);
  // Test that new written size is less than or equal to debug output
  // data bound
  uint32_t obuf_safe_id = TakeNextId();
  AddBinaryOp(GetBoolId(), obuf_safe_id,
      SpvOpULessThanEqual, obuf_new_sz_id, obuf_bnd_id,
      new_blk_ptr);
  uint32_t mergeBlkId = TakeNextId();
  uint32_t writeBlkId = TakeNextId();
  std::unique_ptr<Instruction> mergeLabel(NewLabel(mergeBlkId));
  std::unique_ptr<Instruction> validLabel(NewLabel(writeBlkId));
  AddSelectionMerge(mergeBlkId, SpvSelectionControlMaskNone, new_blk_ptr);
  AddBranchCond(obuf_safe_id, writeBlkId, mergeBlkId, new_blk_ptr);
  // Close safety test block and gen write block
  new_blocks->push_back(std::move(*new_blk_ptr));
  new_blk_ptr->reset(new BasicBlock(std::move(validLabel)));
  GenCommonDebugOutputCode(obuf_record_sz, func_idx,
      instruction_idx, stage_idx, obuf_curr_sz_id,
      new_blk_ptr);
  // TODO(greg-lunarg): Add support for all stages
  uint32_t curr_record_offset = 0;
  if (stage_idx == SpvExecutionModelFragment) {
    GenFragDebugOutputCode(obuf_curr_sz_id, new_blk_ptr);
    curr_record_offset = kInstFragOutRecordSize;
  }
  else {
    assert(false && "unsupported stage");
  }
  // Gen writes of validation specific data
  for (auto vid : validation_data) {
    GenDebugOutputFieldCode(obuf_curr_sz_id, curr_record_offset,
        vid, new_blk_ptr);
    ++curr_record_offset;
  }
  // Close write block and gen merge block
  AddBranch(mergeBlkId, new_blk_ptr);
  new_blocks->push_back(std::move(*new_blk_ptr));
  new_blk_ptr->reset(new BasicBlock(std::move(mergeLabel)));
}

bool InstrumentPass::IsSameBlockOp(const Instruction* inst) const {
  return inst->opcode() == SpvOpSampledImage || inst->opcode() == SpvOpImage;
}

void InstrumentPass::CloneSameBlockOps(
    std::unique_ptr<Instruction>* inst,
    std::unordered_map<uint32_t, uint32_t>* postCallSB,
    std::unordered_map<uint32_t, Instruction*>* preCallSB,
    std::unique_ptr<BasicBlock>* block_ptr) {
  (*inst)->ForEachInId(
      [&postCallSB, &preCallSB, &block_ptr, this](uint32_t* iid) {
        const auto mapItr = (*postCallSB).find(*iid);
        if (mapItr == (*postCallSB).end()) {
          const auto mapItr2 = (*preCallSB).find(*iid);
          if (mapItr2 != (*preCallSB).end()) {
            // Clone pre-call same-block ops, map result id.
            const Instruction* inInst = mapItr2->second;
            std::unique_ptr<Instruction> sb_inst(inInst->Clone(context()));
            CloneSameBlockOps(&sb_inst, postCallSB, preCallSB, block_ptr);
            const uint32_t rid = sb_inst->result_id();
            const uint32_t nid = this->TakeNextId();
            get_decoration_mgr()->CloneDecorations(rid, nid);
            sb_inst->SetResultId(nid);
            (*postCallSB)[rid] = nid;
            *iid = nid;
            (*block_ptr)->AddInstruction(std::move(sb_inst));
          }
        } else {
          // Reset same-block op operand.
          *iid = mapItr->second;
        }
      });
}

void InstrumentPass::UpdateSucceedingPhis(
    std::vector<std::unique_ptr<BasicBlock>>& new_blocks) {
  const auto firstBlk = new_blocks.begin();
  const auto lastBlk = new_blocks.end() - 1;
  const uint32_t firstId = (*firstBlk)->id();
  const uint32_t lastId = (*lastBlk)->id();
  const BasicBlock& const_last_block = *lastBlk->get();
  const_last_block.ForEachSuccessorLabel(
      [&firstId, &lastId, this](const uint32_t succ) {
        BasicBlock* sbp = this->id2block_[succ];
        sbp->ForEachPhiInst([&firstId, &lastId](Instruction* phi) {
          phi->ForEachInId([&firstId, &lastId](uint32_t* id) {
            if (*id == firstId) *id = lastId;
          });
        });
      });
}

// Return id for output buffer uint ptr type
uint32_t InstrumentPass::GetOutputBufferUintPtrId() {
  if (output_buffer_uint_ptr_id_ == 0) {
    output_buffer_uint_ptr_id_ = context()->get_type_mgr()->FindPointerToType(
        GetUintId(), SpvStorageClassStorageBuffer);
  }
  return output_buffer_uint_ptr_id_;
}

uint32_t InstrumentPass::GetOutputBufferBinding() {
  switch (validation_id_) {
    case kInstValidationIdBindless: return kDebugOutputBindingBindless;
    default: assert(false && "unexpected validation id");
  }
  return 0;
}

// Return id for output buffer
uint32_t InstrumentPass::GetOutputBufferId() {
  if (output_buffer_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Integer uint_ty(32, false);
    analysis::Type* reg_uint_ty = type_mgr->GetRegisteredType(&uint_ty);
    analysis::RuntimeArray uint_rarr_ty(reg_uint_ty);
    analysis::Type* reg_uint_rarr_ty = type_mgr->GetRegisteredType(
        &uint_rarr_ty);
    analysis::Struct obuf_ty({ reg_uint_ty, reg_uint_rarr_ty });
    analysis::Type* reg_obuf_ty = type_mgr->GetRegisteredType(&obuf_ty);
    uint32_t obufTyId = type_mgr->GetTypeInstruction(reg_obuf_ty);
    uint32_t obufTyPtrId_ = type_mgr->FindPointerToType(obufTyId,
        SpvStorageClassStorageBuffer);
    output_buffer_id_ = TakeNextId();
    std::unique_ptr<Instruction> newVarOp(
        new Instruction(context(), SpvOpVariable, obufTyPtrId_,
            output_buffer_id_,
            { { spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
                { SpvStorageClassStorageBuffer } } }));
    get_def_use_mgr()->AnalyzeInstDefUse(&*newVarOp);
    get_module()->AddGlobalValue(std::move(newVarOp));
    AddDecoration(output_buffer_id_, SpvDecorationDescriptorSet, desc_set_);
    AddDecoration(output_buffer_id_, SpvDecorationBinding,
        GetOutputBufferBinding());
  }
  return output_buffer_id_;
}

uint32_t InstrumentPass::GetVec4FloatId() {
  if (v4float_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Float float_ty(32);
    analysis::Type* reg_float_ty = type_mgr->GetRegisteredType(&float_ty);
    analysis::Vector v4float_ty(reg_float_ty, 4);
    analysis::Type* reg_v4float_ty = type_mgr->GetRegisteredType(&v4float_ty);
    v4float_id_ = type_mgr->GetTypeInstruction(reg_v4float_ty);
  }
  return v4float_id_;
}

uint32_t InstrumentPass::GetUintId() {
  if (uint_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Integer uint_ty(32, false);
    analysis::Type* reg_uint_ty = type_mgr->GetRegisteredType(&uint_ty);
    uint_id_ = type_mgr->GetTypeInstruction(reg_uint_ty);
  }
  return uint_id_;
}

uint32_t InstrumentPass::GetVec4UintId() {
  if (v4uint_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Integer uint_ty(32, false);
    analysis::Type* reg_uint_ty = type_mgr->GetRegisteredType(&uint_ty);
    analysis::Vector v4uint_ty(reg_uint_ty, 4);
    analysis::Type* reg_v4uint_ty = type_mgr->GetRegisteredType(&v4uint_ty);
    v4uint_id_ = type_mgr->GetTypeInstruction(reg_v4uint_ty);
  }
  return v4uint_id_;
}

uint32_t InstrumentPass::GetBoolId() {
  if (bool_id_ == 0) {
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Bool bool_ty;
    analysis::Type* reg_bool_ty = type_mgr->GetRegisteredType(&bool_ty);
    bool_id_ = type_mgr->GetTypeInstruction(reg_bool_ty);
  }
  return bool_id_;
}

uint32_t InstrumentPass::GetFragCoordId() {
  // If not yet known, look for one in shader
  if (frag_coord_id_ == 0) frag_coord_id_ = FindBuiltin(SpvBuiltInFragCoord);
  // If none in shader, create one
  if (frag_coord_id_ == 0) {
    uint32_t fragCoordTyPtrId = context()->get_type_mgr()->FindPointerToType(
      GetVec4FloatId(), SpvStorageClassInput);
    frag_coord_id_ = TakeNextId();
    std::unique_ptr<Instruction> newVarOp(
      new Instruction(context(), SpvOpVariable, fragCoordTyPtrId,
        frag_coord_id_,
        { { spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
          { SpvStorageClassInput } } }));
    get_def_use_mgr()->AnalyzeInstDefUse(&*newVarOp);
    get_module()->AddGlobalValue(std::move(newVarOp));
    AddDecoration(frag_coord_id_, SpvDecorationBuiltIn, SpvBuiltInFragCoord);
  }
  return frag_coord_id_;
}

bool InstrumentPass::InstProcessCallTreeFromRoots(
    InstProcessFunction& pfn,
    std::queue<uint32_t>* roots,
    uint32_t stage_idx) {
  // Process call tree
  bool modified = false;
  std::unordered_set<uint32_t> done;

  while (!roots->empty()) {
    const uint32_t fi = roots->front();
    roots->pop();
    if (done.insert(fi).second) {
      Function* fn = id2function_.at(fi);
      modified = pfn(fn, stage_idx) || modified;
      AddCalls(fn, roots);
    }
  }
  return modified;
}

bool InstrumentPass::InstProcessEntryPointCallTree(
    InstProcessFunction& pfn,
    Module* module) {
  // Make sure all entry points have the same execution model.
  // TODO(greg-lunarg): Handle mixed stages. Technically, a shader module
  // can contain entry points with different execution models, although
  // such modules will likely be rare as GLSL and HLSL are geared toward
  // one model per module. In such cases we will need
  // to clone any functions which are in the call trees of entrypoints
  // with differing execution models.
  uint32_t ecnt = 0;
  uint32_t eStage;
  for (auto& e : module->entry_points()) {
    if (ecnt == 0)
      eStage = e.GetSingleWordInOperand(kEntryPointExecutionModelInIdx);
    else if (e.GetSingleWordInOperand(kEntryPointExecutionModelInIdx) != eStage)
      return false;
  }
  // Only supporting fragment shaders at the moment.
  // TODO(greg-lunarg): Handle all stages.
  if (eStage != SpvExecutionModelFragment)
    return false;
  // Add together the roots of all entry points
  std::queue<uint32_t> roots;
  for (auto& e : module->entry_points()) {
    roots.push(e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx));
  }
  bool modified = InstProcessCallTreeFromRoots(pfn, &roots,
      SpvExecutionModelFragment);
  // If any function is modified, add FragCoord to all entry points that
  // don't have it.
  if (modified) {
    uint32_t ocnt = 0;
    for (auto& e : module->entry_points()) {
      bool found = false;
      e.ForEachInOperand([&ocnt,&found,this](const uint32_t* idp){
        if (ocnt >= kEntryPointInterfaceInIdx) {
          if (*idp == this->GetFragCoordId()) found = true;
        }
        ++ocnt;
      });
      if (!found) e.AddOperand({ SPV_OPERAND_TYPE_ID, {GetFragCoordId()} });
    }
  }
  return modified;
}

void InstrumentPass::InitializeInstrument(uint32_t validation_id) {
  validation_id_ = validation_id;
  output_buffer_id_ = 0;
  output_buffer_uint_ptr_id_ = 0;
  frag_coord_id_ = 0;
  v4float_id_ = 0;
  uint_id_ = 0;
  v4uint_id_ = 0;
  bool_id_ = 0;

  // clear collections
  id2function_.clear();
  id2block_.clear();

  for (auto& fn : *get_module()) {
    // Initialize function and block maps.
    id2function_[fn.result_id()] = &fn;
    for (auto& blk : fn) {
      id2block_[blk.id()] = &blk;
    }
  }
}

}  // namespace opt
}  // namespace spvtools
