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

#ifndef LIBSPIRV_OPT_INSTRUMENT_PASS_H_
#define LIBSPIRV_OPT_INSTRUMENT_PASS_H_

#include <algorithm>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "decoration_manager.h"
#include "pass.h"

// Validation Ids
static const int kInstValidationIdBindless = 0;

// Error Codes
static const uint32_t kInstErrorBindlessImageBounds = 0;
static const uint32_t kInstErrorBindlessSamplerBounds = 1;
static const uint32_t kInstErrorBindlessImageUninitialized = 2;
static const uint32_t kInstErrorBindlessSamplerUninitialized = 3;

// Debug Buffer Bindings
static const uint32_t kDebugOutputBindingBindless = 0;
static const uint32_t kDebugInputBindingBindless = 1;

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class InstrumentPass : public Pass {
  using cbb_ptr = const BasicBlock*;

 public:
  using GetBlocksFunction =
      std::function<std::vector<BasicBlock*>*(const BasicBlock*)>;

  using InstProcessFunction = std::function<bool(Function*, uint32_t)>;

  virtual ~InstrumentPass() = default;

 protected:
  InstrumentPass(uint32_t desc_set, uint32_t shader_id) :
    desc_set_(desc_set), shader_id_(shader_id) {}

  // Move all code in |ref_block_itr| preceding the instruction |ref_inst_itr|
  // to be instrumented into block |new_blk_ptr|.
  void MovePreludeCode(BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Move all code in |ref_block_itr| succeeding the instruction |ref_inst_itr|
  // to be instrumented into block |new_blk_ptr|.
  void MovePostludeCode(UptrVectorIterator<BasicBlock> ref_block_itr,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Return id for unsigned int constant value |u|.
  uint32_t GetUintConstantId(uint32_t u);

  // Gen code into |new_blk_ptr| to write |field_value_id| into debug output
  // buffer at |base_offset_id| + |field_offset|.
  void GenDebugOutputFieldCode(
    uint32_t base_offset_id,
    uint32_t field_offset,
    uint32_t field_value_id,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Generate instructions into |new_blk_ptr| which will write the members
  // of the debug output buffer common for all stages and validations at
  // |base_off|.
  void GenCommonDebugOutputCode(
    uint32_t record_sz,
    uint32_t func_idx,
    uint32_t instruction_idx,
    uint32_t stage_idx,
    uint32_t base_off,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Generate instructions into |new_blk_ptr| which will write
  // |uint_frag_coord_id| at |component| of the record at |base_off| of
  // the debug output buffer .
  void GenFragCoordEltDebugOutputCode(
    uint32_t base_offset_id,
    uint32_t uint_frag_coord_id,
    uint32_t component,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Generate instructions into |new_blk_ptr| which will write the vertex-
  // shader-specific members of the debug output buffer at |base_off|.
  void GenBuiltinIdOutputCode(
    uint32_t builtinId,
    uint32_t builtinOff,
    uint32_t base_off,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Generate instructions into |new_blk_ptr| which will write the vertex-
  // shader-specific members of the debug output buffer at |base_off|.
  void GenVertDebugOutputCode(
    uint32_t base_off,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Generate instructions into |new_blk_ptr| which will write the fragment-
  // shader-specific members of the debug output buffer at |base_off|.
  void GenFragDebugOutputCode(
    uint32_t base_off,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Return size of common and stage-specific output record members
  uint32_t GetStageOutputRecordSize(uint32_t stage_idx);

  // Generate instructions which will write a record to the end of the debug
  // output buffer for the current shader.
  void GenDebugOutputCode(
    uint32_t func_idx,
    uint32_t instruction_idx,
    uint32_t stage_idx,
    const std::vector<uint32_t> &validation_data,
    std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Add binary instruction |type_id, result_id, opcode, operand1, operand2| to
  // |block_ptr|.
  void AddUnaryOp(
    uint32_t type_id, uint32_t result_id, SpvOp opcode,
    uint32_t operand, std::unique_ptr<BasicBlock>* block_ptr);

  // Add binary instruction |type_id, result_id, opcode, operand1, operand2| to
  // |block_ptr|.
  void AddBinaryOp(
    uint32_t type_id, uint32_t result_id, SpvOp opcode,
    uint32_t operand1, uint32_t operand2,
    std::unique_ptr<BasicBlock>* block_ptr);

  // Add ternary instruction |type_id, result_id, opcode, operand1, operand2,
  // operand3| to |block_ptr|.
  void AddTernaryOp(
    uint32_t type_id, uint32_t result_id, SpvOp opcode,
    uint32_t operand1, uint32_t operand2, uint32_t operand3,
    std::unique_ptr<BasicBlock>* block_ptr);

  // Add quadernary instruction |type_id, result_id, opcode, operand1,
  // operand2, operand3, operand4| to |block_ptr|.
  void AddQuadOp(uint32_t type_id, uint32_t result_id,
    SpvOp opcode, uint32_t operand1, uint32_t operand2, uint32_t operand3,
    uint32_t operand4, std::unique_ptr<BasicBlock>* block_ptr);

  // Add extract instruction |type_id, opcode, operand1, operand2| to
  // |block_ptr| and return resultId.
  void AddExtractOp(
    uint32_t type_id, uint32_t result_id,
    uint32_t operand1, uint32_t operand2,
    std::unique_ptr<BasicBlock>* block_ptr);

  // Add array length instruction |type_id, opcode, struct_ptr_id, member_idx|
  // to |block_ptr| and return resultId.
  void AddArrayLength(uint32_t result_id,
    uint32_t struct_ptr_id, uint32_t member_idx,
    std::unique_ptr<BasicBlock>* block_ptr);

  // Add SelectionMerge instruction |mergeBlockId, selectionControl| to
  // |block_ptr|.
  void AddSelectionMerge(
    uint32_t mergeBlockId, uint32_t selectionControl,
    std::unique_ptr<BasicBlock>* block_ptr);

  // Return id of pointer to builtin |builtin_val|.
  uint32_t FindBuiltin(uint32_t builtin_val);

  // Add |decoration, decoration_value| of |inst_id| to module. Also
  // update decoration manager.
  void AddDecoration(uint32_t inst_id, uint32_t decoration);
  
  // Add |decoration, decoration_value| of |inst_id| to module. Also
  // update decoration manager.
  void AddDecorationVal(uint32_t inst_id, uint32_t decoration,
    uint32_t decoration_value);

  // Add |decoration, decoration_value| of |inst_id| to module. Also
  // update decoration manager.
  void AddMemberDecoration(uint32_t member, uint32_t inst_id,
    uint32_t decoration, uint32_t decoration_value);

  // Add unconditional branch to labelId to end of block block_ptr.
  void AddBranch(uint32_t labelId, std::unique_ptr<BasicBlock>* block_ptr);

  // Add conditional branch to end of block |block_ptr|.
  void AddBranchCond(uint32_t cond_id, uint32_t true_id, uint32_t false_id,
                     std::unique_ptr<BasicBlock>* block_ptr);

  // Add Phi to |block_ptr|.
  void AddPhi(uint32_t type_id, uint32_t result_id, uint32_t var0_id,
              uint32_t parent0_id, uint32_t var1_id, uint32_t parent1_id,
              std::unique_ptr<BasicBlock>* block_ptr);

  // Return new label.
  std::unique_ptr<Instruction> NewLabel(uint32_t label_id);

  // Returns the id for the null constant value of |type_id|.
  uint32_t GetNullId(uint32_t type_id);

  // Return true if instruction must be in the same block that its result
  // is used.
  bool IsSameBlockOp(const Instruction* inst) const;

  // Clone operands which must be in same block as consumer instructions.
  // Look in preCallSB for instructions that need cloning. Look in
  // postCallSB for instructions already cloned. Add cloned instruction
  // to postCallSB.
  void CloneSameBlockOps(std::unique_ptr<Instruction>* inst,
                         std::unordered_map<uint32_t, uint32_t>* postCallSB,
                         std::unordered_map<uint32_t, Instruction*>* preCallSB,
                         std::unique_ptr<BasicBlock>* block_ptr);

  // Update phis in succeeding blocks to point to new last block
  void UpdateSucceedingPhis(
      std::vector<std::unique_ptr<BasicBlock>>& new_blocks);

  // Return id for 32-bit unsigned type
  uint32_t GetUintId();

  // Return id for 32-bit unsigned type
  uint32_t GetBoolId();

  // Return id for output buffer uint type
  uint32_t GetOutputBufferUintPtrId();
  
  // Return binding for output buffer for current validation.
  uint32_t GetOutputBufferBinding();

  // Return id for debug output buffer
  uint32_t GetOutputBufferId();

  // Return id for variable with |builtin| decoration. Create if it
  // doesn't exist.
  uint32_t GetBuiltinVarId(uint32_t builtin, uint32_t type_id, uint32_t* var_id);

  // Return id for VertexId variable
  uint32_t GetVertexId();

  // Return id for InstanceId variable
  uint32_t GetInstanceId();

  // Return id for FragCoord variable
  uint32_t GetFragCoordId();

  // Return id for v4float type
  uint32_t GetVec4FloatId();

  // Return id for v4uint type
  uint32_t GetVec4UintId();

  // Add |var_id| to all entry points if not there.
  void AddVarToEntryPoints(uint32_t var_id);
  
  // Call |pfn| on all functions in the call tree of the function
  // ids in |roots|. 
  bool InstProcessCallTreeFromRoots(
    InstProcessFunction& pfn,
    std::queue<uint32_t>* roots,
    uint32_t stage_idx);

  // Call |pfn| on all functions in the call tree of the entry points
  // in |module|.
  bool InstProcessEntryPointCallTree(
    InstProcessFunction& pfn,
    Module* module);

  // Initialize state for optimization of module
  void InitializeInstrument(uint32_t validation_id);

  // Debug descriptor set index
  uint32_t desc_set_;

  // Shader module ID written into output record
  uint32_t shader_id_;

  // Map from function id to function pointer.
  std::unordered_map<uint32_t, Function*> id2function_;

  // Map from block's label id to block. TODO(dnovillo): This is superfluous wrt
  // CFG. It has functionality not present in CFG. Consolidate.
  std::unordered_map<uint32_t, BasicBlock*> id2block_;

  // result id for OpConstantFalse
  uint32_t validation_id_;

  // id for output buffer variable
  uint32_t output_buffer_id_;

  // type id for output buffer element
  uint32_t output_buffer_uint_ptr_id_;

  // id for Vertex
  uint32_t vertex_id_;

  // id for Instance
  uint32_t instance_id_;

  // id for FragCoord
  uint32_t frag_coord_id_;

  // id for v4float type
  uint32_t v4float_id_;

  // id for v4float type
  uint32_t v4uint_id_;

  // id for 32-bit unsigned type
  uint32_t uint_id_;

  // id for bool type
  uint32_t bool_id_;

  // Pre-instrumentation same-block insts
  std::unordered_map<uint32_t, Instruction*> preCallSB_;

  // Post-instrumentation same-block op ids
  std::unordered_map<uint32_t, uint32_t> postCallSB_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INSTRUMENT_PASS_H_
