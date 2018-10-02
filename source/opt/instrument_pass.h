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
#include "ir_builder.h"
#include "pass.h"

namespace spvtools {
namespace opt {

// Validation Ids
static const int kInstValidationIdBindless = 0;

// Error Codes
static const uint32_t kInstErrorBindlessBounds = 0;
static const uint32_t kInstErrorBindlessUninitialized = 1;

// Debug Buffer Bindings
static const uint32_t kDebugOutputBindingBindless = 0;
static const uint32_t kDebugInputBindingBindless = 1;

// Preserved Analyses
static const IRContext::Analysis kInstPreservedAnalyses = 
    IRContext::kAnalysisDefUse;

// This is a base class to assist in the creation of passes which instrument
// modules. More specifically, passes which replace instructions with a larger
// and more capable set of instructions. Commonly, these new instructions will
// add testing of operands and execute different instructions depending on the
// outcome, including outputting of a debug record into a buffer created
// especially for that purpose.
//
// This class contains helper functions to create an InstProcessFunction,
// which is the heart of any derived class implementing a specific
// instrumentation pass. It takes an instruction as an argument, decides
// if it should be instrumented, and generates code to replace it. This class
// also supplies function InstProcessEntryPointCallTree which applies the
// InstProcessFunction to every reachable instruction in a module and replaces
// the instruction with new instructions if generated.
//
// One of the main helper functions supplied is GenDebugOutputCode, which
// generates code to write a debug output record along with code to test if
// space remains in the debug output buffer. The record contains three
// subsections: members common across all validation, members specific to
// the stage and members specific to a validation. These are enumerated
// using static const offsets in the .cpp file.

class InstrumentPass : public Pass {
  using cbb_ptr = const BasicBlock*;

 public:
  using InstProcessFunction = std::function<void(
    BasicBlock::iterator,
    UptrVectorIterator<BasicBlock>,
    uint32_t,
    uint32_t,
    uint32_t,
    std::vector<std::unique_ptr<BasicBlock>>*)>;

  virtual ~InstrumentPass() = default;

 protected:
  // Create instrumentation pass which utilizes descriptor set |desc_set|
  // for debug input and output buffers and writes |shader_id| into debug
  // output records.
  InstrumentPass(uint32_t desc_set, uint32_t shader_id,
      uint32_t validation_id) : desc_set_(desc_set), shader_id_(shader_id),
      validation_id_(validation_id) {}

  // Initialize state for instrumentation of module by |validation_id|.
  void InitializeInstrument();

  // Call |pfn| on all instructions in all functions in the call tree of the
  // entry points in |module|. If code is generated for an instruction, replace
  // the instruction's block with the new blocks that are generated. Continue
  // processing at the top of the last new block.
  bool InstProcessEntryPointCallTree(
    InstProcessFunction& pfn,
    Module* module);

  // Move all code in |ref_block_itr| preceding the instruction |ref_inst_itr|
  // to be instrumented into block |new_blk_ptr|.
  void MovePreludeCode(BasicBlock::iterator ref_inst_itr,
    UptrVectorIterator<BasicBlock> ref_block_itr,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Move all code in |ref_block_itr| succeeding the instruction |ref_inst_itr|
  // to be instrumented into block |new_blk_ptr|.
  void MovePostludeCode(UptrVectorIterator<BasicBlock> ref_block_itr,
    std::unique_ptr<BasicBlock>* new_blk_ptr);

  // Generate instructions which will write a record to the end of the debug
  // output buffer for the current shader if enough space remains.
  void GenDebugOutputCode(
    uint32_t func_idx,
    uint32_t instruction_idx,
    uint32_t stage_idx,
    const std::vector<uint32_t> &validation_ids,
    InstructionBuilder* builder);

  // Return id for unsigned int constant value |u|.
  uint32_t GetUintConstantId(uint32_t u);

  // Generate code to cast |value_id| to unsigned, if needed. Return
  // an id to the unsigned equivalent.
  uint32_t GenUintCastCode(
    uint32_t value_id,
    InstructionBuilder* builder);

  // Return new label.
  std::unique_ptr<Instruction> NewLabel(uint32_t label_id);

  // Returns the id for the null constant value of |type_id|.
  uint32_t GetNullId(uint32_t type_id);

  // Return id for 32-bit unsigned type
  uint32_t GetUintId();

  // Return id for 32-bit unsigned type
  uint32_t GetBoolId();

  // Return id for void type
  uint32_t GetVoidId();

  // Return id for output buffer uint type
  uint32_t GetOutputBufferUintPtrId();
  
  // Return binding for output buffer for current validation.
  uint32_t GetOutputBufferBinding();

  // Return id for debug output buffer
  uint32_t GetOutputBufferId();

  // Return id for v4float type
  uint32_t GetVec4FloatId();

  // Return id for v4uint type
  uint32_t GetVec4UintId();

  // Return id for output function. Define if it doesn't exist with
  // |val_spec_arg_cnt| validation-specific uint32 arguments.
  uint32_t GetOutputFunctionId(uint32_t stage_idx,
      uint32_t val_spec_param_cnt);

  // Add |var_id| to all entry points if not there.
  void AddVarToEntryPoints(uint32_t var_id);

  // Apply instrumentation function |pfn| to every instruction in |func|.
  // If code is generated for an instruction, replace the instruction's
  // block with the new blocks that are generated. Continue processing at the
  // top of the last new block.
  bool InstrumentFunction(Function* func, uint32_t stage_idx,
      InstProcessFunction& pfn);
  
  // Call |pfn| on all functions in the call tree of the function
  // ids in |roots|. 
  bool InstProcessCallTreeFromRoots(
    InstProcessFunction& pfn,
    std::queue<uint32_t>* roots,
    uint32_t stage_idx);

  // Gen code into |new_blk_ptr| to write |field_value_id| into debug output
  // buffer at |base_offset_id| + |field_offset|.
  void GenDebugOutputFieldCode(
    uint32_t base_offset_id,
    uint32_t field_offset,
    uint32_t field_value_id,
    InstructionBuilder* builder);

  // Generate instructions into |new_blk_ptr| which will write the members
  // of the debug output record common for all stages and validations at
  // |base_off|.
  void GenCommonDebugOutputCode(
    uint32_t record_sz,
    uint32_t func_idx,
    uint32_t instruction_idx,
    uint32_t stage_idx,
    uint32_t base_off,
    InstructionBuilder* builder);

  // Generate instructions into |new_blk_ptr| which will write
  // |uint_frag_coord_id| at |component| of the record at |base_offset_id| of
  // the debug output buffer .
  void GenFragCoordEltDebugOutputCode(
    uint32_t base_offset_id,
    uint32_t uint_frag_coord_id,
    uint32_t component,
    InstructionBuilder* builder);

  // Generate instructions into |new_blk_ptr| which will write the vertex-
  // shader-specific members of the debug output buffer at |base_off|.
  void GenBuiltinIdOutputCode(
    uint32_t builtin_id,
    uint32_t builtin_off,
    uint32_t base_off,
    InstructionBuilder* builder);

  // Generate instructions into |new_blk_ptr| which will write the vertex-
  // shader-specific members of the debug output buffer at |base_off|.
  void GenVertDebugOutputCode(
    uint32_t base_off,
    InstructionBuilder* builder);

  // Generate instructions into |new_blk_ptr| which will write the fragment-
  // shader-specific members of the debug output buffer at |base_off|.
  void GenFragDebugOutputCode(
    uint32_t base_off,
    InstructionBuilder* builder);

  // Return size of common and stage-specific output record members
  uint32_t GetStageOutputRecordSize(uint32_t stage_idx);

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

  // id for debug output function
  uint32_t output_func_id_;

  // param count for output function
  uint32_t output_func_param_cnt_;

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
