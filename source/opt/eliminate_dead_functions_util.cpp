// Copyright (c) 2019 Google LLC
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

#include "eliminate_dead_functions_util.h"

namespace spvtools {
namespace opt {

namespace eliminatedeadfunctionsutil {

Pass::Status EliminateFunction(IRContext* context,
                               Module::iterator* func_iter) {
  bool first_func = *func_iter == context->module()->begin();
  bool seen_func_end = false;
  std::unordered_set<Instruction*> to_kill;
  std::vector<Instruction*> insts_to_kill;
  bool failure = false;
  (*func_iter)
      ->ForEachInst(
          [context, first_func, func_iter, &seen_func_end, &to_kill,
           &failure, &insts_to_kill](Instruction* inst) {
            if (failure) return;
            if (inst->opcode() == spv::Op::OpFunctionEnd) {
              seen_func_end = true;
            }
            // Move non-semantic instructions to the previous function or
            // global values if this is the first function.
            if (seen_func_end && inst->opcode() == spv::Op::OpExtInst) {
              assert(inst->IsNonSemanticInstruction());
              if (to_kill.find(inst) != to_kill.end()) return;
              std::unique_ptr<Instruction> clone(inst->Clone(context));
              if (!clone) {
                failure = true;
                return;
              }
              // Clear uses of "inst" to in case this moves a dependent chain of
              // instructions.
              context->get_def_use_mgr()->ClearInst(inst);
              context->AnalyzeDefUse(clone.get());
              if (first_func) {
                context->AddGlobalValue(std::move(clone));
              } else {
                auto prev_func_iter = *func_iter;
                --prev_func_iter;
                prev_func_iter->AddNonSemanticInstruction(std::move(clone));
              }
              inst->ToNop();
            } else if (to_kill.find(inst) == to_kill.end()) {
              context->CollectNonSemanticTree(inst, &to_kill);
              insts_to_kill.push_back(inst);
            }
          },
          true, true);

  if (failure) return Pass::Status::Failure;

  for (auto* dead : to_kill) {
    context->KillInst(dead);
  }
  for (auto* dead : insts_to_kill) {
    context->KillInst(dead);
  }

  *func_iter = func_iter->Erase();
  return Pass::Status::SuccessWithChange;
}

}  // namespace eliminatedeadfunctionsutil
}  // namespace opt
}  // namespace spvtools
