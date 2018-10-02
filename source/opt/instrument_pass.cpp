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

// Common Parameter Positions
static const int kInstCommonParamFuncIdx = 0;
static const int kInstCommonParamInstIdx = 1;
static const int kInstCommonParamCnt = 2;

// Vertex Shader Output Record Offsets
static const int kInstVertOutVertexId = 5;
static const int kInstVertOutInstanceId = 6;
static const int kInstVertOutRecordSize = 7;

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

uint32_t InstrumentPass::GenUintCastCode(
    uint32_t val_id,
    InstructionBuilder* builder) {
  // Cast value to 32-bit unsigned if necessary
  if (get_def_use_mgr()->GetDef(val_id)->type_id() == GetUintId())
    return val_id;
  return builder->AddUnaryOp(GetUintId(), SpvOpBitcast, val_id)->result_id();
}

void InstrumentPass::GenDebugOutputFieldCode(
    uint32_t base_offset_id,
    uint32_t field_offset,
    uint32_t field_value_id,
    InstructionBuilder* builder) {
  // Cast value to 32-bit unsigned if necessary
  uint32_t val_id = GenUintCastCode(field_value_id, builder);
  // Store value
  Instruction* data_idx_inst = builder->AddBinaryOp(GetUintId(), SpvOpIAdd,
      base_offset_id, GetUintConstantId(field_offset));
  Instruction* achain_inst = builder->AddTernaryOp(GetOutputBufferUintPtrId(),
      SpvOpAccessChain, GetOutputBufferId(),
      GetUintConstantId(kDebugOutputDataOffset), data_idx_inst->result_id());
  (void) builder->AddBinaryOp(0, SpvOpStore, achain_inst->result_id(), val_id);
}

void InstrumentPass::GenCommonDebugOutputCode(
    uint32_t record_sz,
    uint32_t func_id,
    uint32_t inst_id,
    uint32_t stage_idx,
    uint32_t base_offset_id,
    InstructionBuilder* builder) {
  // Store record size
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutSize,
      GetUintConstantId(record_sz), builder);
  // Store Shader Id
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutShaderId,
      GetUintConstantId(shader_id_), builder);
  // Store Function Idx
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutFunctionIdx,
      func_id, builder);
  // Store Instruction Idx
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutInstructionIdx,
      inst_id, builder);
  // Store Stage Idx
  GenDebugOutputFieldCode(base_offset_id, kInstCommonOutStageIdx,
      GetUintConstantId(stage_idx), builder);
}

void InstrumentPass::GenFragCoordEltDebugOutputCode(
    uint32_t base_offset_id,
    uint32_t uint_frag_coord_id,
    uint32_t element,
  InstructionBuilder* builder) {
  Instruction* element_val_inst = builder->AddIdLiteralOp(GetUintId(),
      SpvOpCompositeExtract, uint_frag_coord_id, element);
  GenDebugOutputFieldCode(base_offset_id, kInstFragOutFragCoordX + element,
      element_val_inst->result_id(), builder);
}

void InstrumentPass::GenBuiltinIdOutputCode(
    uint32_t builtinId,
    uint32_t builtinOff,
    uint32_t base_offset_id,
    InstructionBuilder* builder) {
  // Load and store builtin
  Instruction* load_inst = builder->AddUnaryOp(GetUintId(), SpvOpLoad, builtinId);
  GenDebugOutputFieldCode(base_offset_id, builtinOff, load_inst->result_id(),
      builder);
}

void InstrumentPass::GenVertDebugOutputCode(
    uint32_t base_offset_id,
    InstructionBuilder* builder) {
  // Load and store VertexId and InstanceId
  GenBuiltinIdOutputCode(context()->GetBuiltinVarId(SpvBuiltInVertexId),
      kInstVertOutVertexId, base_offset_id, builder);
  GenBuiltinIdOutputCode(context()->GetBuiltinVarId(SpvBuiltInInstanceId),
      kInstVertOutInstanceId, base_offset_id, builder);
}

void InstrumentPass::GenFragDebugOutputCode(
    uint32_t base_offset_id,
    InstructionBuilder* builder) {
  // Load FragCoord and convert to Uint
  Instruction* frag_coord_inst = builder->AddUnaryOp(GetVec4FloatId(),
      SpvOpLoad, context()->GetBuiltinVarId(SpvBuiltInFragCoord));
  Instruction* uint_frag_coord_inst = builder->AddUnaryOp(GetVec4UintId(),
      SpvOpBitcast, frag_coord_inst->result_id());
  for (uint32_t u = 0; u < 2u; ++u)
    GenFragCoordEltDebugOutputCode(base_offset_id,
        uint_frag_coord_inst->result_id(), u, builder);
}

uint32_t InstrumentPass::GetStageOutputRecordSize(uint32_t stage_idx) {
  // TODO(greg-lunarg): Add support for all stages
  uint32_t size = 0;
  switch (stage_idx) {
    case SpvExecutionModelVertex:   size = kInstVertOutRecordSize; break;
    case SpvExecutionModelFragment: size = kInstFragOutRecordSize; break;
    default: assert(false && "unexpected stage"); break;
  }
  return size;
}

void InstrumentPass::GenDebugOutputCode(
    uint32_t func_idx,
    uint32_t instruction_idx,
    uint32_t stage_idx,
    const std::vector<uint32_t> &validation_ids,
    InstructionBuilder* builder) {
  // Call debug output function. Pass func_idx, instruction_idx and
  // validation ids as args.
  uint32_t val_id_cnt = static_cast<uint32_t>(validation_ids.size());
  uint32_t output_func_id = GetOutputFunctionId(stage_idx, val_id_cnt);
  std::vector<uint32_t> args = { output_func_id, GetUintConstantId(func_idx),
      GetUintConstantId(instruction_idx) };
  (void) args.insert(args.end(), validation_ids.begin(), validation_ids.end());
  (void) builder->AddNaryOp(GetVoidId(), SpvOpFunctionCall, args);
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
    analysis::DecorationManager* deco_mgr = context()->get_decoration_mgr();
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    analysis::Integer uint_ty(32, false);
    analysis::Type* reg_uint_ty = type_mgr->GetRegisteredType(&uint_ty);
    analysis::RuntimeArray uint_rarr_ty(reg_uint_ty);
    analysis::Type* reg_uint_rarr_ty = type_mgr->GetRegisteredType(
        &uint_rarr_ty);
    analysis::Struct obuf_ty({ reg_uint_ty, reg_uint_rarr_ty });
    analysis::Type* reg_obuf_ty = type_mgr->GetRegisteredType(&obuf_ty);
    uint32_t obufTyId = type_mgr->GetTypeInstruction(reg_obuf_ty);
    deco_mgr->AddDecoration(obufTyId, SpvDecorationBlock);
    deco_mgr->AddMemberDecoration(obufTyId, kDebugOutputSizeOffset,
        SpvDecorationOffset, 0);
    deco_mgr->AddMemberDecoration(obufTyId, kDebugOutputDataOffset,
        SpvDecorationOffset, 4);
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
    deco_mgr->AddDecorationVal(output_buffer_id_, SpvDecorationDescriptorSet,
        desc_set_);
    deco_mgr->AddDecorationVal(output_buffer_id_, SpvDecorationBinding,
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

uint32_t InstrumentPass::GetVoidId() {
  analysis::TypeManager* type_mgr = context()->get_type_mgr();
  analysis::Void void_ty;
  analysis::Type* reg_void_ty = type_mgr->GetRegisteredType(&void_ty);
  return type_mgr->GetTypeInstruction(reg_void_ty);
}

uint32_t InstrumentPass::GetOutputFunctionId(uint32_t stage_idx,
    uint32_t val_spec_param_cnt) {
  // Total param count is common params plus validation-specific
  // params
  uint32_t param_cnt = kInstCommonParamCnt + val_spec_param_cnt;
  if (output_func_id_ == 0) {
    // Create function
    output_func_id_ = TakeNextId();
    analysis::TypeManager* type_mgr = context()->get_type_mgr();
    std::vector<const analysis::Type*> param_types;
    for (uint32_t c = 0; c < param_cnt; ++c)
      param_types.push_back(type_mgr->GetType(GetUintId()));
    analysis::Function func_ty(type_mgr->GetType(GetVoidId()), param_types);
    analysis::Type* reg_func_ty = type_mgr->GetRegisteredType(&func_ty);
    std::unique_ptr<Instruction> func_inst(
        new Instruction(get_module()->context(), SpvOpFunction, GetVoidId(),
            output_func_id_, 
            { { spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
            { SpvFunctionControlMaskNone } },
            { spv_operand_type_t::SPV_OPERAND_TYPE_ID,
            { type_mgr->GetTypeInstruction(reg_func_ty) } } }));
    get_def_use_mgr()->AnalyzeInstDefUse(&*func_inst);
    std::unique_ptr<Function> output_func = MakeUnique<Function>(
        std::move(func_inst));
    // Add parameters
    std::vector<uint32_t> param_vec;
    for (uint32_t c = 0; c < param_cnt; ++c) {
      uint32_t pid = TakeNextId();
      param_vec.push_back(pid);
      std::unique_ptr<Instruction> param_inst(
          new Instruction(get_module()->context(), SpvOpFunctionParameter,
              GetUintId(), pid, {}));
      get_def_use_mgr()->AnalyzeInstDefUse(&*param_inst);
      output_func->AddParameter(std::move(param_inst));
    }
    // Create first block
    uint32_t test_blk_id = TakeNextId();
    std::unique_ptr<Instruction> test_label(NewLabel(test_blk_id));
    std::unique_ptr<BasicBlock> new_blk_ptr = MakeUnique<BasicBlock>(
        std::move(test_label));
    InstructionBuilder builder(context(), &*new_blk_ptr, kInstPreservedAnalyses);
    // Gen test if debug output buffer size will not be exceeded.
    uint32_t obuf_record_sz = GetStageOutputRecordSize(stage_idx) +
        val_spec_param_cnt;
    Instruction* obuf_curr_sz_ac_inst = builder.AddBinaryOp(
        GetOutputBufferUintPtrId(),
        SpvOpAccessChain, GetOutputBufferId(),
        GetUintConstantId(kDebugOutputSizeOffset));
    // Fetch the current debug buffer written size atomically, adding the
    // size of the record to be written.
    Instruction* obuf_curr_sz_inst = builder.AddQuadOp(GetUintId(),
        SpvOpAtomicIAdd,
        obuf_curr_sz_ac_inst->result_id(),
        GetUintConstantId(SpvScopeInvocation),
        GetUintConstantId(SpvMemoryAccessMaskNone),
        GetUintConstantId(obuf_record_sz));
    uint32_t obuf_curr_sz_id = obuf_curr_sz_inst->result_id();
    // Compute new written size
    Instruction* obuf_new_sz_inst = builder.AddBinaryOp(GetUintId(), SpvOpIAdd,
        obuf_curr_sz_id, GetUintConstantId(obuf_record_sz));
    // Fetch the data bound
    Instruction* obuf_bnd_inst = builder.AddIdLiteralOp(GetUintId(),
        SpvOpArrayLength, GetOutputBufferId(), kDebugOutputDataOffset);
    // Test that new written size is less than or equal to debug output
    // data bound
    Instruction* obuf_safe_inst = builder.AddBinaryOp(GetBoolId(),
        SpvOpULessThanEqual, obuf_new_sz_inst->result_id(),
        obuf_bnd_inst->result_id());
    uint32_t mergeBlkId = TakeNextId();
    uint32_t writeBlkId = TakeNextId();
    std::unique_ptr<Instruction> mergeLabel(NewLabel(mergeBlkId));
    std::unique_ptr<Instruction> writeLabel(NewLabel(writeBlkId));
    (void) builder.AddConditionalBranch(obuf_safe_inst->result_id(),
        writeBlkId, mergeBlkId, mergeBlkId, SpvSelectionControlMaskNone);
    // Close safety test block and gen write block
    new_blk_ptr->SetParent(&*output_func);
    output_func->AddBasicBlock(std::move(new_blk_ptr));
    new_blk_ptr = MakeUnique<BasicBlock>(std::move(writeLabel));
    builder.SetInsertPoint(&*new_blk_ptr);
    // Generate common debug record members
    GenCommonDebugOutputCode(obuf_record_sz,
        param_vec[kInstCommonParamFuncIdx], param_vec[kInstCommonParamInstIdx],
        stage_idx, obuf_curr_sz_id, &builder);
    // Generate stage-specific record members
    // TODO(greg-lunarg): Add support for all stages
    uint32_t stage_offset = 0;
    switch (stage_idx) {
    case SpvExecutionModelFragment:
      GenFragDebugOutputCode(obuf_curr_sz_id, &builder);
      stage_offset = kInstFragOutRecordSize;
      break;
    case SpvExecutionModelVertex:
      GenVertDebugOutputCode(obuf_curr_sz_id, &builder);
      stage_offset = kInstVertOutRecordSize;
      break;
    default:
      assert(false && "unsupported stage");
      break;
    }
    // Gen writes of validation specific data
    for (uint32_t i = 0; i < val_spec_param_cnt; ++i) {
      GenDebugOutputFieldCode(obuf_curr_sz_id, stage_offset + i,
          param_vec[kInstCommonOutFunctionIdx + i], &builder);
    }
    // Close write block and gen merge block
    (void) builder.AddBranch(mergeBlkId);
    new_blk_ptr->SetParent(&*output_func);
    output_func->AddBasicBlock(std::move(new_blk_ptr));
    new_blk_ptr = MakeUnique<BasicBlock>(std::move(mergeLabel));
    builder.SetInsertPoint(&*new_blk_ptr);
    // Close merge block and function and add function to module
    (void) builder.AddNullaryOp(0, SpvOpReturn);
    new_blk_ptr->SetParent(&*output_func);
    output_func->AddBasicBlock(std::move(new_blk_ptr));
    std::unique_ptr<Instruction> func_end_inst(
        new Instruction(get_module()->context(), SpvOpFunctionEnd,
            0, 0, {}));
    get_def_use_mgr()->AnalyzeInstDefUse(&*func_end_inst);
    output_func->SetFunctionEnd(std::move(func_end_inst));
    get_module()->AddFunction(std::move(output_func));
    output_func_param_cnt_ = param_cnt;
  }
  assert(param_cnt == output_func_param_cnt_ && "bad arg count");
  return output_func_id_;
}

void InstrumentPass::AddVarToEntryPoints(uint32_t var_id) {
  uint32_t ocnt = 0;
  for (auto& e : get_module()->entry_points()) {
    bool found = false;
    e.ForEachInOperand([&ocnt, &found, &var_id](const uint32_t* idp) {
      if (ocnt >= kEntryPointInterfaceInIdx) {
        if (*idp == var_id) found = true;
      }
      ++ocnt;
    });
    if (!found) {
      e.AddOperand({ SPV_OPERAND_TYPE_ID,{ var_id } });
      get_def_use_mgr()->AnalyzeInstDefUse(&e);
    }
  }
}

bool InstrumentPass::InstrumentFunction(Function* func, uint32_t stage_idx,
    InstProcessFunction& pfn) {
  bool modified = false;
  // Compute function index
  uint32_t function_idx = 0;
  for (auto fii = get_module()->begin(); fii != get_module()->end(); ++fii) {
    if (&*fii == func)
      break;
    ++function_idx;
  }
  std::vector<std::unique_ptr<BasicBlock>> newBlocks;
  // Count function instruction
  uint32_t instruction_idx = 1;
  // Using block iterators here because of block erasures and insertions.
  for (auto bi = func->begin(); bi != func->end(); ++bi) {
    // Count block's label
    ++instruction_idx;
    for (auto ii = bi->begin(); ii != bi->end(); ++instruction_idx) {
      // Bump instruction count if debug instructions
      instruction_idx += static_cast<uint32_t>(ii->dbg_line_insts().size());
      // Generate bindless check if warranted
      pfn(ii, bi, function_idx, instruction_idx, stage_idx, &newBlocks);
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
      modified = true;
      // Restart instrumenting at beginning of last new block,
      // but skip over any new phi or copy instruction.
      ii = bi->begin();
      if (ii->opcode() == SpvOpPhi || ii->opcode() == SpvOpCopyObject) ++ii;
      newBlocks.clear();
    }
  }
  return modified;
}

bool InstrumentPass::InstProcessCallTreeFromRoots(
    InstProcessFunction& pfn,
    std::queue<uint32_t>* roots,
    uint32_t stage_idx) {
  // Process call tree
  bool modified = false;
  std::unordered_set<uint32_t> done;

  // Process all functions from roots
  while (!roots->empty()) {
    const uint32_t fi = roots->front();
    roots->pop();
    if (done.insert(fi).second) {
      Function* fn = id2function_.at(fi);
      // Add calls first so we don't add new output function
      AddCalls(fn, roots);
      modified = InstrumentFunction(fn, stage_idx, pfn) || modified;
    }
  }
  return modified;
}

bool InstrumentPass::InstProcessEntryPointCallTree(
    InstProcessFunction& pfn,
    Module* module) {
  // Make sure all entry points have the same execution model. Do not
  // instrument if they do not.
  // TODO(greg-lunarg): Handle mixed stages. Technically, a shader module
  // can contain entry points with different execution models, although
  // such modules will likely be rare as GLSL and HLSL are geared toward
  // one model per module. In such cases we will need
  // to clone any functions which are in the call trees of entrypoints
  // with differing execution models.
  uint32_t ecnt = 0;
  uint32_t eStage;
  for (auto& e : get_module()->entry_points()) {
    if (ecnt == 0)
      eStage = e.GetSingleWordInOperand(kEntryPointExecutionModelInIdx);
    else if (e.GetSingleWordInOperand(kEntryPointExecutionModelInIdx) != eStage)
      return false;
  }
  // Only supporting vertex and fragment shaders at the moment.
  // TODO(greg-lunarg): Handle all stages.
  if (eStage != SpvExecutionModelVertex &&
      eStage != SpvExecutionModelFragment)
    return false;
  // Add together the roots of all entry points
  std::queue<uint32_t> roots;
  for (auto& e : get_module()->entry_points()) {
    roots.push(e.GetSingleWordInOperand(kEntryPointFunctionIdInIdx));
  }
  bool modified = InstProcessCallTreeFromRoots(pfn, &roots, eStage);
  // Add builtins to all entry points that don't have them.
  if (frag_coord_id_ != 0) AddVarToEntryPoints(frag_coord_id_);
  if (vertex_id_ != 0) AddVarToEntryPoints(vertex_id_);
  if (instance_id_ != 0) AddVarToEntryPoints(instance_id_);
  return modified;
}

void InstrumentPass::InitializeInstrument() {
  output_buffer_id_ = 0;
  output_buffer_uint_ptr_id_ = 0;
  output_func_id_ = 0;
  output_func_param_cnt_ = 0;
  v4float_id_ = 0;
  uint_id_ = 0;
  v4uint_id_ = 0;
  bool_id_ = 0;
  vertex_id_ = 0;
  instance_id_ = 0;
  frag_coord_id_ = 0;

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
