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

#ifndef LIBSPIRV_OPT_INST_BINDLESS_CHECK_PASS_H_
#define LIBSPIRV_OPT_INST_BINDLESS_CHECK_PASS_H_

#include <algorithm>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

#include "def_use_manager.h"
#include "instrument_pass.h"
#include "module.h"

namespace spvtools {
namespace opt {

// See optimizer.hpp for documentation.
class InstBindlessCheckPass : public InstrumentPass {
 public:
   InstBindlessCheckPass(uint32_t desc_set, uint32_t shader_id)
     : InstrumentPass(desc_set, shader_id) {}

  Status Process() override;

  const char* name() const override { return "inline-entry-points-exhaustive"; }

 private:
   // Initialize state for instrumenting bindless checking
   void InitializeInstBindlessCheck();

   // If |ref_inst_itr| is a bindless reference, return in |new_blocks| the
   // result of instrumenting it with validation code within its block at
   // |ref_block_itr|. Specifically, generate code to check that the index
   // into the descriptor array is in-bounds. If the check passes, execute
   // the remainder of the reference, otherwise write a record to the debug
   // output buffer and replace the reference with 0. The block at
   // |ref_block_itr| can just be replaced with the blocks in |new_blocks|,
   // which will contain at least two blocks. The last block will
   // contain all instructions following the instruction being instrumented,
   // preceded by either a phi or copyobject instruction.
   // Note that the first block in |new_blocks| retains the label
   // of the original calling block.
   void GenBindlessCheckCode(std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
     BasicBlock::iterator ref_inst_itr,
     UptrVectorIterator<BasicBlock> ref_block_itr,
     uint32_t function_idx,
     uint32_t instruction_idx,
     uint32_t stage_idx);

  Pass::Status ProcessImpl();

  // True if VK_EXT_descriptor_indexing is defined
  bool ext_descriptor_indexing_defined_;
};

}  // namespace opt
}  // namespace spvtools

#endif  // LIBSPIRV_OPT_INST_BINDLESS_CHECK_PASS_H_
