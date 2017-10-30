// Copyright (c) 2017 Google Inc.
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

#ifndef SPIRV_TOOLS_DEAD_VARIABLE_ELIMINATION_H
#define SPIRV_TOOLS_DEAD_VARIABLE_ELIMINATION_H

#include <unordered_map>
#include <climits>

#include "decoration_manager.h"
#include "mem_pass.h"

namespace spvtools {
namespace opt {

class DeadVariableElimination : public MemPass {
 public:
  const char* name() const override { return "dead-variable-elimination"; }
  Status Process(ir::IRContext* c) override;

 private:
  // Deletes the OpVariable instruction who result id is |result_id|.
  void DeleteVariable(uint32_t result_id);

  // Keeps track of the number of references of an id.  Once that value is 0, it
  // is safe to remove the corresponding instruction.
  //
  // Note that the special value kMustKeep is used to indicate that the
  // instruction cannot be deleted for reasons other that is being explicitly
  // referenced.
  std::unordered_map<uint32_t, size_t> reference_count_;

  // Special value used to indicate that an id cannot be safely deleted.
  enum { kMustKeep = INT_MAX };
};

}  // namespace opt
}  // namespace spvtools

#endif  // SPIRV_TOOLS_DEAD_VARIABLE_ELIMINATION_H
