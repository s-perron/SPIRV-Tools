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

void InstBindlessCheckPass::GenBindlessCheckCode(std::vector<std::unique_ptr<BasicBlock>>* new_blocks,
  std::vector<std::unique_ptr<Instruction>>* new_vars,
  BasicBlock::iterator ref_inst_itr,
  UptrVectorIterator<BasicBlock> ref_block_itr) {
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
        // If call block is replaced with more than one block, point
        // succeeding phis at new last block.
        if (newBlocks.size() > 1) UpdateSucceedingPhis(newBlocks);
        // Replace old calling block with new block(s).

        // We need to kill the name and decorations for the call, which
        // will be deleted.  Other instructions in the block will be moved to
        // newBlocks.  We don't need to do anything with those.
        context()->KillNamesAndDecorates(&*ii);

        bi = bi.Erase();

        for (auto& bb : newBlocks) {
          bb->SetParent(func);
        }
        bi = bi.InsertBefore(&newBlocks);
        // Insert new function variables.
        if (newVars.size() > 0)
          func->begin()->begin().InsertBefore(std::move(newVars));
        // Restart inlining at beginning of calling block.
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
