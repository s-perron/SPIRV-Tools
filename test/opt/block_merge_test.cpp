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

#include <string>

#include "test/opt/pass_fixture.h"
#include "test/opt/pass_utils.h"

namespace spvtools {
namespace opt {
namespace {

using BlockMergeTest = PassTest<::testing::Test>;

TEST_F(BlockMergeTest, Simple) {
  // Note: SPIR-V hand edited to insert block boundary
  // between two statements in main.
  //
  //  #version 140
  //
  //  in vec4 BaseColor;
  //
  //  void main()
  //  {
  //      vec4 v = BaseColor;
  //      gl_FragColor = v;
  //  }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%13 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%14 = OpLoad %v4float %BaseColor
OpStore %v %14
OpBranch %15
%15 = OpLabel
%16 = OpLoad %v4float %v
OpStore %gl_FragColor %16
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%13 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%14 = OpLoad %v4float %BaseColor
OpStore %v %14
%16 = OpLoad %v4float %v
OpStore %gl_FragColor %16
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<BlockMergePass>(predefs + before, predefs + after, true,
                                        true);
}

TEST_F(BlockMergeTest, BlockMergeForLinkage) {
  const std::string before =
      R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpSource HLSL 630
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %bb_entry "bb.entry"
OpName %v "v"
OpDecorate %main LinkageAttributes "main" Export
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%8 = OpTypeFunction %v4float %_ptr_Function_v4float
%main = OpFunction %v4float None %8
%BaseColor = OpFunctionParameter %_ptr_Function_v4float
%bb_entry = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%9 = OpLoad %v4float %BaseColor
OpStore %v %9
OpBranch %10
%10 = OpLabel
%11 = OpLoad %v4float %v
OpBranch %12
%12 = OpLabel
OpReturnValue %11
OpFunctionEnd
)";

  const std::string after =
      R"(OpCapability Shader
OpCapability Linkage
OpMemoryModel Logical GLSL450
OpSource HLSL 630
OpName %main "main"
OpName %BaseColor "BaseColor"
OpName %bb_entry "bb.entry"
OpName %v "v"
OpDecorate %main LinkageAttributes "main" Export
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%8 = OpTypeFunction %v4float %_ptr_Function_v4float
%main = OpFunction %v4float None %8
%BaseColor = OpFunctionParameter %_ptr_Function_v4float
%bb_entry = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%9 = OpLoad %v4float %BaseColor
OpStore %v %9
%11 = OpLoad %v4float %v
OpReturnValue %11
OpFunctionEnd
)";
  SinglePassRunAndCheck<BlockMergePass>(before, after, true, true);
}

TEST_F(BlockMergeTest, EmptyBlock) {
  // Note: SPIR-V hand edited to insert empty block
  // after two statements in main.
  //
  //  #version 140
  //
  //  in vec4 BaseColor;
  //
  //  void main()
  //  {
  //      vec4 v = BaseColor;
  //      gl_FragColor = v;
  //  }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %gl_FragColor "gl_FragColor"
%void = OpTypeVoid
%7 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %7
%13 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%14 = OpLoad %v4float %BaseColor
OpStore %v %14
OpBranch %15
%15 = OpLabel
%16 = OpLoad %v4float %v
OpStore %gl_FragColor %16
OpBranch %17
%17 = OpLabel
OpBranch %18
%18 = OpLabel
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %7
%13 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%14 = OpLoad %v4float %BaseColor
OpStore %v %14
%16 = OpLoad %v4float %v
OpStore %gl_FragColor %16
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<BlockMergePass>(predefs + before, predefs + after, true,
                                        true);
}

TEST_F(BlockMergeTest, NestedInControlFlow) {
  // Note: SPIR-V hand edited to insert block boundary
  // between OpFMul and OpStore in then-part.
  //
  // #version 140
  // in vec4 BaseColor;
  //
  // layout(std140) uniform U_t
  // {
  //     bool g_B ;
  // } ;
  //
  // void main()
  // {
  //     vec4 v = BaseColor;
  //     if (g_B)
  //       vec4 v = v * 0.25;
  //     gl_FragColor = v;
  // }

  const std::string predefs =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %BaseColor %gl_FragColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %v "v"
OpName %BaseColor "BaseColor"
OpName %U_t "U_t"
OpMemberName %U_t 0 "g_B"
OpName %_ ""
OpName %v_0 "v"
OpName %gl_FragColor "gl_FragColor"
OpMemberDecorate %U_t 0 Offset 0
OpDecorate %U_t Block
OpDecorate %_ DescriptorSet 0
%void = OpTypeVoid
%10 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Function_v4float = OpTypePointer Function %v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%uint = OpTypeInt 32 0
%U_t = OpTypeStruct %uint
%_ptr_Uniform_U_t = OpTypePointer Uniform %U_t
%_ = OpVariable %_ptr_Uniform_U_t Uniform
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%_ptr_Uniform_uint = OpTypePointer Uniform %uint
%bool = OpTypeBool
%uint_0 = OpConstant %uint 0
%float_0_25 = OpConstant %float 0.25
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
)";

  const std::string before =
      R"(%main = OpFunction %void None %10
%24 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%v_0 = OpVariable %_ptr_Function_v4float Function
%25 = OpLoad %v4float %BaseColor
OpStore %v %25
%26 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%27 = OpLoad %uint %26
%28 = OpINotEqual %bool %27 %uint_0
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
%31 = OpLoad %v4float %v
%32 = OpVectorTimesScalar %v4float %31 %float_0_25
OpBranch %33
%33 = OpLabel
OpStore %v_0 %32
OpBranch %29
%29 = OpLabel
%34 = OpLoad %v4float %v
OpStore %gl_FragColor %34
OpReturn
OpFunctionEnd
)";

  const std::string after =
      R"(%main = OpFunction %void None %10
%24 = OpLabel
%v = OpVariable %_ptr_Function_v4float Function
%v_0 = OpVariable %_ptr_Function_v4float Function
%25 = OpLoad %v4float %BaseColor
OpStore %v %25
%26 = OpAccessChain %_ptr_Uniform_uint %_ %int_0
%27 = OpLoad %uint %26
%28 = OpINotEqual %bool %27 %uint_0
OpSelectionMerge %29 None
OpBranchConditional %28 %30 %29
%30 = OpLabel
%31 = OpLoad %v4float %v
%32 = OpVectorTimesScalar %v4float %31 %float_0_25
OpStore %v_0 %32
OpBranch %29
%29 = OpLabel
%34 = OpLoad %v4float %v
OpStore %gl_FragColor %34
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<BlockMergePass>(predefs + before, predefs + after, true,
                                        true);
}

TEST_F(BlockMergeTest, PhiInSuccessorOfMergedBlock) {
  const std::string text = R"(
; CHECK: OpSelectionMerge [[merge:%\w+]] None
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[then:%\w+]] [[else:%\w+]]
; CHECK: [[then]] = OpLabel
; CHECK-NEXT: OpBranch [[merge]]
; CHECK: [[else]] = OpLabel
; CHECK-NEXT: OpBranch [[merge]]
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: OpPhi {{%\w+}} %true [[then]] %false [[else]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse  %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%entry = OpLabel
OpSelectionMerge %merge None
OpBranchConditional %true %then %else
%then = OpLabel
OpBranch %then_next
%then_next = OpLabel
OpBranch %merge
%else = OpLabel
OpBranch %merge
%merge = OpLabel
%phi = OpPhi %bool %true %then_next %false %else
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, UpdateMergeInstruction) {
  const std::string text = R"(
; CHECK: OpSelectionMerge [[merge:%\w+]] None
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[then:%\w+]] [[else:%\w+]]
; CHECK: [[then]] = OpLabel
; CHECK-NEXT: OpBranch [[merge]]
; CHECK: [[else]] = OpLabel
; CHECK-NEXT: OpBranch [[merge]]
; CHECK: [[merge]] = OpLabel
; CHECK-NEXT: OpReturn
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse  %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%entry = OpLabel
OpSelectionMerge %real_merge None
OpBranchConditional %true %then %else
%then = OpLabel
OpBranch %merge
%else = OpLabel
OpBranch %merge
%merge = OpLabel
OpBranch %real_merge
%real_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, TwoMergeBlocksCannotBeMerged) {
  const std::string text = R"(
; CHECK: OpSelectionMerge [[outer_merge:%\w+]] None
; CHECK: OpSelectionMerge [[inner_merge:%\w+]] None
; CHECK: [[inner_merge]] = OpLabel
; CHECK-NEXT: OpBranch [[outer_merge]]
; CHECK: [[outer_merge]] = OpLabel
; CHECK-NEXT: OpReturn
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse  %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%entry = OpLabel
OpSelectionMerge %outer_merge None
OpBranchConditional %true %then %else
%then = OpLabel
OpBranch %inner_header
%else = OpLabel
OpBranch %inner_header
%inner_header = OpLabel
OpSelectionMerge %inner_merge None
OpBranchConditional %true %inner_then %inner_else
%inner_then = OpLabel
OpBranch %inner_merge
%inner_else = OpLabel
OpBranch %inner_merge
%inner_merge = OpLabel
OpBranch %outer_merge
%outer_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, MergeContinue) {
  const std::string text = R"(
; CHECK: OpBranch [[header:%\w+]]
; CHECK: [[header]] = OpLabel
; CHECK-NEXT: OpLogicalAnd
; CHECK-NEXT: OpLoopMerge {{%\w+}} [[header]] None
; CHECK-NEXT: OpBranch [[header]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse  %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%entry = OpLabel
OpBranch %header
%header = OpLabel
OpLoopMerge %merge %continue None
OpBranch %continue
%continue = OpLabel
%op = OpLogicalAnd %bool %true %false
OpBranch %header
%merge = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, MergeContinueWithOpLine) {
  const std::string text = R"(
; CHECK: OpBranch [[header:%\w+]]
; CHECK: [[header]] = OpLabel
; CHECK-NEXT: OpLogicalAnd
; CHECK-NEXT: OpLine {{%\w+}} 1 1
; CHECK-NEXT: OpLoopMerge {{%\w+}} [[header]] None
; CHECK-NEXT: OpBranch [[header]]
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%src = OpString "test.shader"
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse  %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%entry = OpLabel
OpBranch %header
%header = OpLabel
OpLoopMerge %merge %continue None
OpBranch %continue
%continue = OpLabel
%op = OpLogicalAnd %bool %true %false
OpLine %src 1 1
OpBranch %header
%merge = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, TwoHeadersCannotBeMerged) {
  const std::string text = R"(
; CHECK: OpBranch [[loop_header:%\w+]]
; CHECK: [[loop_header]] = OpLabel
; CHECK-NEXT: OpLoopMerge
; CHECK-NEXT: OpBranch [[if_header:%\w+]]
; CHECK: [[if_header]] = OpLabel
; CHECK-NEXT: OpSelectionMerge
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse  %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%entry = OpLabel
OpBranch %header
%header = OpLabel
OpLoopMerge %merge %continue None
OpBranch %inner_header
%inner_header = OpLabel
OpSelectionMerge %if_merge None
OpBranchConditional %true %then %if_merge
%then = OpLabel
OpBranch %continue
%if_merge = OpLabel
OpBranch %continue
%continue = OpLabel
OpBranchConditional %false %merge %header
%merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, CannotMergeContinue) {
  const std::string text = R"(
; CHECK: OpBranch [[loop_header:%\w+]]
; CHECK: [[loop_header]] = OpLabel
; CHECK-NEXT: OpLoopMerge {{%\w+}} [[continue:%\w+]]
; CHECK-NEXT: OpBranchConditional {{%\w+}} [[if_header:%\w+]]
; CHECK: [[if_header]] = OpLabel
; CHECK-NEXT: OpSelectionMerge
; CHECK: [[continue]] = OpLabel
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%true = OpConstantTrue %bool
%false = OpConstantFalse  %bool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%entry = OpLabel
OpBranch %header
%header = OpLabel
OpLoopMerge %merge %continue None
OpBranchConditional %true %inner_header %merge
%inner_header = OpLabel
OpSelectionMerge %if_merge None
OpBranchConditional %true %then %if_merge
%then = OpLabel
OpBranch %continue
%if_merge = OpLabel
OpBranch %continue
%continue = OpLabel
OpBranchConditional %false %merge %header
%merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, RemoveStructuredDeclaration) {
  // Note: SPIR-V hand edited remove dead branch and add block
  // before continue block
  //
  // #version 140
  // in vec4 BaseColor;
  //
  // void main()
  // {
  //     while (true) {
  //         break;
  //     }
  //     gl_FragColor = BaseColor;
  // }

  const std::string assembly =
      R"(
; CHECK: OpLabel
; CHECK: [[header:%\w+]] = OpLabel
; CHECK-NOT: OpLoopMerge
; CHECK: OpReturn
; CHECK: [[continue:%\w+]] = OpLabel
; CHECK-NEXT: OpBranch [[block:%\w+]]
; CHECK: [[block]] = OpLabel
; CHECK-NEXT: OpBranch [[header]]
OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %gl_FragColor %BaseColor
OpExecutionMode %main OriginUpperLeft
OpSource GLSL 140
OpName %main "main"
OpName %gl_FragColor "gl_FragColor"
OpName %BaseColor "BaseColor"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%bool = OpTypeBool
%true = OpConstantTrue %bool
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%gl_FragColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
%BaseColor = OpVariable %_ptr_Input_v4float Input
%main = OpFunction %void None %6
%13 = OpLabel
OpBranch %14
%14 = OpLabel
OpLoopMerge %15 %16 None
OpBranch %17
%17 = OpLabel
OpBranch %15
%18 = OpLabel
OpBranch %16
%16 = OpLabel
OpBranch %14
%15 = OpLabel
%19 = OpLoad %v4float %BaseColor
OpStore %gl_FragColor %19
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(assembly, true);
}

TEST_F(BlockMergeTest, DontMergeKill) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] [[cont:%\w+]] None
; CHECK-NEXT: OpBranch [[ret:%\w+]]
; CHECK: [[ret:%\w+]] = OpLabel
; CHECK-NEXT: OpKill
; CHECK-DAG: [[cont]] = OpLabel
; CHECK-DAG: [[merge]] = OpLabel
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %3 %4 None
OpBranch %5
%5 = OpLabel
OpKill
%4 = OpLabel
OpBranch %2
%3 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, DontMergeTerminateInvocation) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] [[cont:%\w+]] None
; CHECK-NEXT: OpBranch [[ret:%\w+]]
; CHECK: [[ret:%\w+]] = OpLabel
; CHECK-NEXT: OpTerminateInvocation
; CHECK-DAG: [[cont]] = OpLabel
; CHECK-DAG: [[merge]] = OpLabel
OpCapability Shader
OpExtension "SPV_KHR_terminate_invocation"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %3 %4 None
OpBranch %5
%5 = OpLabel
OpTerminateInvocation
%4 = OpLabel
OpBranch %2
%3 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, DontMergeUnreachable) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] [[cont:%\w+]] None
; CHECK-NEXT: OpBranch [[ret:%\w+]]
; CHECK: [[ret:%\w+]] = OpLabel
; CHECK-NEXT: OpUnreachable
; CHECK-DAG: [[cont]] = OpLabel
; CHECK-DAG: [[merge]] = OpLabel
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %3 %4 None
OpBranch %5
%5 = OpLabel
OpUnreachable
%4 = OpLabel
OpBranch %2
%3 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, false);
}

TEST_F(BlockMergeTest, DontMergeReturn) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] [[cont:%\w+]] None
; CHECK-NEXT: OpBranch [[ret:%\w+]]
; CHECK: [[ret:%\w+]] = OpLabel
; CHECK-NEXT: OpReturn
; CHECK-DAG: [[cont]] = OpLabel
; CHECK-DAG: [[merge]] = OpLabel
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %3 %4 None
OpBranch %5
%5 = OpLabel
OpReturn
%4 = OpLabel
OpBranch %2
%3 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, DontMergeSwitch) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] [[cont:%\w+]] None
; CHECK-NEXT: OpBranch [[ret:%\w+]]
; CHECK: [[ret:%\w+]] = OpLabel
; CHECK-NEXT: OpSelectionMerge
; CHECK-NEXT: OpSwitch
; CHECK-DAG: [[cont]] = OpLabel
; CHECK-DAG: [[merge]] = OpLabel
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%int = OpTypeInt 32 1
%int_0 = OpConstant %int 0
%functy = OpTypeFunction %void
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranch %2
%2 = OpLabel
OpLoopMerge %3 %4 None
OpBranch %5
%5 = OpLabel
OpSelectionMerge %6 None
OpSwitch %int_0 %6
%6 = OpLabel
OpReturn
%4 = OpLabel
OpBranch %2
%3 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, DontMergeReturnValue) {
  const std::string text = R"(
; CHECK: OpLoopMerge [[merge:%\w+]] [[cont:%\w+]] None
; CHECK-NEXT: OpBranch [[ret:%\w+]]
; CHECK: [[ret:%\w+]] = OpLabel
; CHECK-NEXT: OpReturn
; CHECK-DAG: [[cont]] = OpLabel
; CHECK-DAG: [[merge]] = OpLabel
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%functy = OpTypeFunction %void
%otherfuncty = OpTypeFunction %bool
%true = OpConstantTrue %bool
%func = OpFunction %void None %functy
%1 = OpLabel
%2 = OpFunctionCall %bool %3
OpReturn
OpFunctionEnd
%3 = OpFunction %bool None %otherfuncty
%4 = OpLabel
OpBranch %5
%5 = OpLabel
OpLoopMerge %6 %7 None
OpBranch %8
%8 = OpLabel
OpReturnValue %true
%7 = OpLabel
OpBranch %5
%6 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, MergeHeaders) {
  // Merge two headers when the second is the merge block of the first.
  const std::string text = R"(
; CHECK: OpFunction
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpBranch [[header:%\w+]]
; CHECK-NEXT: [[header]] = OpLabel
; CHECK-NEXT: OpSelectionMerge [[merge:%\w+]]
; CHECK: [[merge]] = OpLabel
; CHECK: OpReturn
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %func "func"
OpExecutionMode %func OriginUpperLeft
%void = OpTypeVoid
%bool = OpTypeBool
%functy = OpTypeFunction %void
%otherfuncty = OpTypeFunction %bool
%true = OpConstantTrue %bool
%func = OpFunction %void None %functy
%1 = OpLabel
OpBranch %5
%5 = OpLabel
OpLoopMerge %8 %7 None
OpBranch %8
%7 = OpLabel
OpBranch %5
%8 = OpLabel
OpSelectionMerge %m None
OpBranchConditional %true %a %m
%a = OpLabel
OpBranch %m
%m = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, OpPhiInSuccessor) {
  // Checks that when merging blocks A and B, the OpPhi at the start of B is
  // removed and uses of its definition are replaced appropriately.
  const std::string prefix =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource ESSL 310
OpName %main "main"
OpName %x "x"
OpName %y "y"
%void = OpTypeVoid
%6 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%int_1 = OpConstant %int 1
%main = OpFunction %void None %6
%10 = OpLabel
%x = OpVariable %_ptr_Function_int Function
%y = OpVariable %_ptr_Function_int Function
OpStore %x %int_1
%11 = OpLoad %int %x
)";

  const std::string suffix_before =
      R"(OpBranch %12
%12 = OpLabel
%13 = OpPhi %int %11 %10
OpStore %y %13
OpReturn
OpFunctionEnd
)";

  const std::string suffix_after =
      R"(OpStore %y %11
OpReturn
OpFunctionEnd
)";
  SinglePassRunAndCheck<BlockMergePass>(prefix + suffix_before,
                                        prefix + suffix_after, true, true);
}

TEST_F(BlockMergeTest, MultipleOpPhisInSuccessor) {
  // Checks that when merging blocks A and B, the OpPhis at the start of B are
  // removed and uses of their definitions are replaced appropriately.
  const std::string prefix =
      R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource ESSL 310
OpName %main "main"
OpName %S "S"
OpMemberName %S 0 "x"
OpMemberName %S 1 "f"
OpName %s "s"
OpName %g "g"
OpName %y "y"
OpName %t "t"
OpName %z "z"
%void = OpTypeVoid
%10 = OpTypeFunction %void
%int = OpTypeInt 32 1
%float = OpTypeFloat 32
%S = OpTypeStruct %int %float
%_ptr_Function_S = OpTypePointer Function %S
%int_1 = OpConstant %int 1
%float_2 = OpConstant %float 2
%16 = OpConstantComposite %S %int_1 %float_2
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_Function_int = OpTypePointer Function %int
%int_3 = OpConstant %int 3
%int_0 = OpConstant %int 0
%main = OpFunction %void None %10
%21 = OpLabel
%s = OpVariable %_ptr_Function_S Function
%g = OpVariable %_ptr_Function_float Function
%y = OpVariable %_ptr_Function_int Function
%t = OpVariable %_ptr_Function_S Function
%z = OpVariable %_ptr_Function_float Function
OpStore %s %16
OpStore %g %float_2
OpStore %y %int_3
%22 = OpLoad %S %s
OpStore %t %22
%23 = OpAccessChain %_ptr_Function_float %s %int_1
%24 = OpLoad %float %23
%25 = OpLoad %float %g
)";

  const std::string suffix_before =
      R"(OpBranch %26
%26 = OpLabel
%27 = OpPhi %float %24 %21
%28 = OpPhi %float %25 %21
%29 = OpFAdd %float %27 %28
%30 = OpAccessChain %_ptr_Function_int %s %int_0
%31 = OpLoad %int %30
OpBranch %32
%32 = OpLabel
%33 = OpPhi %float %29 %26
%34 = OpPhi %int %31 %26
%35 = OpConvertSToF %float %34
OpBranch %36
%36 = OpLabel
%37 = OpPhi %float %35 %32
%38 = OpFSub %float %33 %37
%39 = OpLoad %int %y
OpBranch %40
%40 = OpLabel
%41 = OpPhi %float %38 %36
%42 = OpPhi %int %39 %36
%43 = OpConvertSToF %float %42
%44 = OpFAdd %float %41 %43
OpStore %z %44
OpReturn
OpFunctionEnd
)";

  const std::string suffix_after =
      R"(%29 = OpFAdd %float %24 %25
%30 = OpAccessChain %_ptr_Function_int %s %int_0
%31 = OpLoad %int %30
%35 = OpConvertSToF %float %31
%38 = OpFSub %float %29 %35
%39 = OpLoad %int %y
%43 = OpConvertSToF %float %39
%44 = OpFAdd %float %38 %43
OpStore %z %44
OpReturn
OpFunctionEnd
)";
  SinglePassRunAndCheck<BlockMergePass>(prefix + suffix_before,
                                        prefix + suffix_after, true, true);
}

TEST_F(BlockMergeTest, UnreachableLoop) {
  const std::string spirv = R"(OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource ESSL 310
OpName %main "main"
%void = OpTypeVoid
%4 = OpTypeFunction %void
%int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
%bool = OpTypeBool
%false = OpConstantFalse %bool
%main = OpFunction %void None %4
%9 = OpLabel
OpBranch %10
%11 = OpLabel
OpLoopMerge %12 %13 None
OpBranchConditional %false %13 %14
%13 = OpLabel
OpSelectionMerge %15 None
OpBranchConditional %false %16 %17
%16 = OpLabel
OpBranch %15
%17 = OpLabel
OpBranch %15
%15 = OpLabel
OpBranch %11
%14 = OpLabel
OpReturn
%12 = OpLabel
OpBranch %10
%10 = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndCheck<BlockMergePass>(spirv, spirv, true, true);
}

TEST_F(BlockMergeTest, DebugMerge) {
  // Verify merge can be done completely, cleanly and validly in presence of
  // NonSemantic.Shader.DebugInfo.100 instructions
  const std::string text = R"(
; CHECK: OpLoopMerge
; CHECK-NEXT: OpBranch
; CHECK-NOT: OpBranch
OpCapability Shader
OpExtension "SPV_KHR_non_semantic_info"
%1 = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %in_var_COLOR %out_var_SV_TARGET
OpExecutionMode %main OriginUpperLeft
%5 = OpString "lexblock.hlsl"
%20 = OpString "float"
%32 = OpString "main"
%33 = OpString ""
%46 = OpString "b"
%49 = OpString "a"
%58 = OpString "c"
%63 = OpString "color"
OpName %in_var_COLOR "in.var.COLOR"
OpName %out_var_SV_TARGET "out.var.SV_TARGET"
OpName %main "main"
OpDecorate %in_var_COLOR Location 0
OpDecorate %out_var_SV_TARGET Location 0
%float = OpTypeFloat 32
%float_0 = OpConstant %float 0
%v4float = OpTypeVector %float 4
%9 = OpConstantComposite %v4float %float_0 %float_0 %float_0 %float_0
%float_1 = OpConstant %float 1
%13 = OpConstantComposite %v4float %float_1 %float_1 %float_1 %float_1
%uint = OpTypeInt 32 0
%uint_32 = OpConstant %uint 32
%_ptr_Input_v4float = OpTypePointer Input %v4float
%_ptr_Output_v4float = OpTypePointer Output %v4float
%void = OpTypeVoid
%uint_3 = OpConstant %uint 3
%uint_0 = OpConstant %uint 0
%uint_4 = OpConstant %uint 4
%uint_1 = OpConstant %uint 1
%uint_5 = OpConstant %uint 5
%uint_12 = OpConstant %uint 12
%uint_13 = OpConstant %uint 13
%uint_20 = OpConstant %uint 20
%uint_15 = OpConstant %uint 15
%uint_17 = OpConstant %uint 17
%uint_16 = OpConstant %uint 16
%uint_14 = OpConstant %uint 14
%uint_10 = OpConstant %uint 10
%65 = OpTypeFunction %void
%in_var_COLOR = OpVariable %_ptr_Input_v4float Input
%out_var_SV_TARGET = OpVariable %_ptr_Output_v4float Output
%62 = OpExtInst %void %1 DebugExpression
%22 = OpExtInst %void %1 DebugTypeBasic %20 %uint_32 %uint_3 %uint_0
%25 = OpExtInst %void %1 DebugTypeVector %22 %uint_4
%27 = OpExtInst %void %1 DebugTypeFunction %uint_3 %25 %25
%28 = OpExtInst %void %1 DebugSource %5
%29 = OpExtInst %void %1 DebugCompilationUnit %uint_1 %uint_4 %28 %uint_5
%34 = OpExtInst %void %1 DebugFunction %32 %27 %28 %uint_12 %uint_1 %29 %33 %uint_3 %uint_13
%37 = OpExtInst %void %1 DebugLexicalBlock %28 %uint_13 %uint_1 %34
%52 = OpExtInst %void %1 DebugLexicalBlock %28 %uint_15 %uint_12 %37
%54 = OpExtInst %void %1 DebugLocalVariable %46 %25 %28 %uint_17 %uint_12 %52 %uint_4
%56 = OpExtInst %void %1 DebugLocalVariable %49 %25 %28 %uint_16 %uint_12 %52 %uint_4
%59 = OpExtInst %void %1 DebugLocalVariable %58 %25 %28 %uint_14 %uint_10 %37 %uint_4
%64 = OpExtInst %void %1 DebugLocalVariable %63 %25 %28 %uint_12 %uint_20 %34 %uint_4 %uint_1
%main = OpFunction %void None %65
%66 = OpLabel
%69 = OpLoad %v4float %in_var_COLOR
%168 = OpExtInst %void %1 DebugValue %64 %69 %62
%169 = OpExtInst %void %1 DebugScope %37
OpLine %5 14 10
%164 = OpExtInst %void %1 DebugValue %59 %9 %62
OpLine %5 15 3
OpBranch %150
%150 = OpLabel
%165 = OpPhi %v4float %9 %66 %158 %159
%167 = OpExtInst %void %1 DebugValue %59 %165 %62
%170 = OpExtInst %void %1 DebugScope %37
OpLine %5 15 12
%171 = OpExtInst %void %1 DebugNoScope
OpLoopMerge %160 %159 None
OpBranch %151
%151 = OpLabel
OpLine %5 16 12
%162 = OpExtInst %void %1 DebugValue %56 %9 %62
OpLine %5 17 12
%163 = OpExtInst %void %1 DebugValue %54 %13 %62
OpLine %5 18 15
%158 = OpFAdd %v4float %165 %13
OpLine %5 18 5
%166 = OpExtInst %void %1 DebugValue %59 %158 %62
%172 = OpExtInst %void %1 DebugScope %37
OpLine %5 19 3
OpBranch %159
%159 = OpLabel
OpLine %5 19 3
OpBranch %150
%160 = OpLabel
OpUnreachable
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, DontLoseCaseConstruct) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %func "func"
OpExecutionMode %func LocalSize 1 1 1
OpName %entry "entry";
OpName %loop "loop"
OpName %loop_merge "loop_merge"
OpName %loop_cont "loop_cont"
OpName %switch "switch"
OpName %switch_merge "switch_merge"
%void = OpTypeVoid
%bool = OpTypeBool
%int = OpTypeInt 32 0
%void_fn = OpTypeFunction %void
%bool_undef = OpUndef %bool
%int_undef = OpUndef %int
%func = OpFunction %void None %void_fn
%entry = OpLabel
OpBranch %loop
%loop = OpLabel
OpLoopMerge %loop_merge %loop_cont None
OpBranch %switch
%switch = OpLabel
OpSelectionMerge %switch_merge None
OpSwitch %int_undef %switch_merge 0 %break 1 %break
%break = OpLabel
OpBranch %loop_merge
%switch_merge = OpLabel
OpBranch %loop_cont
%loop_cont = OpLabel
OpBranch %loop
%loop_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<opt::BlockMergePass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(opt::Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(BlockMergeTest, DontLoseDefaultCaseConstruct) {
  const std::string text = R"(
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %func "func"
OpExecutionMode %func LocalSize 1 1 1
OpName %entry "entry";
OpName %loop "loop"
OpName %loop_merge "loop_merge"
OpName %loop_cont "loop_cont"
OpName %switch "switch"
OpName %switch_merge "switch_merge"
%void = OpTypeVoid
%bool = OpTypeBool
%int = OpTypeInt 32 0
%void_fn = OpTypeFunction %void
%bool_undef = OpUndef %bool
%int_undef = OpUndef %int
%func = OpFunction %void None %void_fn
%entry = OpLabel
OpBranch %loop
%loop = OpLabel
OpLoopMerge %loop_merge %loop_cont None
OpBranch %switch
%switch = OpLabel
OpSelectionMerge %switch_merge None
OpSwitch %int_undef %cont 0 %switch_merge 1 %switch_merge
%cont = OpLabel
OpBranch %loop_cont
%switch_merge = OpLabel
OpBranch %loop_merge
%loop_cont = OpLabel
OpBranch %loop
%loop_merge = OpLabel
OpReturn
OpFunctionEnd
)";

  auto result = SinglePassRunAndDisassemble<opt::BlockMergePass>(
      text, /* skip_nop = */ true, /* do_validation = */ true);
  EXPECT_EQ(opt::Pass::Status::SuccessWithoutChange, std::get<1>(result));
}

TEST_F(BlockMergeTest, RebuildStructuredCFG) {
  const std::string text = R"(
; CHECK: = OpFunction
; CHECK-NEXT: [[entry:%\w+]] = OpLabel
; CHECK-NEXT: OpSelectionMerge [[merge:%\w+]] None
; CHECK-NEXT: OpSwitch {{%\w+}} [[merge]] 0 [[other:%\w+]]
; CHECK [[other]] = OpLabel
; CHECK: OpBranch [[merge]]
; CHECK [[merge]] = OpLabel
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%void = OpTypeVoid
%int = OpTypeInt 32 0
%int_1 = OpConstant %int 1
%void_fn = OpTypeFunction %void
%main = OpFunction %void None %void_fn
%entry = OpLabel
OpBranch %switch
%switch = OpLabel
OpSelectionMerge %merge None
OpSwitch %int_1 %merge 0 %other
%other = OpLabel
OpBranch %merge
%merge = OpLabel
OpReturn
OpFunctionEnd
)";

  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, MaximalReconvergenceNoMeldToMerge) {
  const std::string text = R"(
                              OpCapability Shader
                              OpCapability GroupNonUniformBallot
                              OpCapability GroupNonUniformArithmetic
                              OpExtension "SPV_KHR_maximal_reconvergence"
                              OpMemoryModel Logical GLSL450
                              OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID %output
                              OpExecutionMode %main LocalSize 1 1 1
                              OpExecutionMode %main MaximallyReconvergesKHR
                              OpSource HLSL 660
                              OpName %type_RWStructuredBuffer_uint "type.RWStructuredBuffer.uint"
                              OpName %output "output"
                              OpName %main "main"
                              OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
                              OpDecorate %output DescriptorSet 0
                              OpDecorate %output Binding 0
                              OpDecorate %_runtimearr_uint ArrayStride 4
                              OpMemberDecorate %type_RWStructuredBuffer_uint 0 Offset 0
                              OpDecorate %type_RWStructuredBuffer_uint Block
                      %uint = OpTypeInt 32 0
                      %bool = OpTypeBool
                       %int = OpTypeInt 32 1
                     %int_0 = OpConstant %int 0
                     %int_1 = OpConstant %int 1
               %_runtimearr_uint = OpTypeRuntimeArray %uint
               %type_RWStructuredBuffer_uint = OpTypeStruct %_runtimearr_uint
               %_ptr_StorageBuffer_type_RWStructuredBuffer_uint = OpTypePointer StorageBuffer %type_RWStructuredBuffer_uint
                    %v3uint = OpTypeVector %uint 3
               %_ptr_Input_v3uint = OpTypePointer Input %v3uint
                      %void = OpTypeVoid
                        %15 = OpTypeFunction %void
                    %uint_3 = OpConstant %uint 3
               %_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
                    %output = OpVariable %_ptr_StorageBuffer_type_RWStructuredBuffer_uint StorageBuffer
               %gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
                      %main = OpFunction %void None %15
                        %18 = OpLabel
                        %19 = OpLoad %v3uint %gl_GlobalInvocationID
                              OpBranch %20
                        %20 = OpLabel
                              OpLoopMerge %21 %22 None
; CHECK:                      OpLoopMerge [[merge:%\w+]] [[continue:%\w+]]
                              OpBranch %23
                        %23 = OpLabel
                        %24 = OpCompositeExtract %uint %19 0
                        %25 = OpGroupNonUniformBroadcastFirst %uint %uint_3 %24
                        %26 = OpIEqual %bool %24 %25
                              OpSelectionMerge %27 None
                              OpBranchConditional %26 %28 %27
                        %28 = OpLabel
                        %29 = OpGroupNonUniformIAdd %int %uint_3 Reduce %int_1
                        %30 = OpBitcast %uint %29
                              OpBranch %21
; CHECK:        [[t1:%\w+]] = OpGroupNonUniformIAdd %int %uint_3 Reduce %int_1
; CHECK-NEXT:   [[t2:%\w+]] = OpBitcast %uint [[t1]]
; CHECK-NEXT:                 OpBranch [[merge]]
                        %27 = OpLabel
                              OpBranch %22
                        %22 = OpLabel
                              OpBranch %20
                        %21 = OpLabel
                        %31 = OpAccessChain %_ptr_StorageBuffer_uint %output %int_0 %24
                              OpStore %31 %30
                              OpReturn
                              OpFunctionEnd
)";

  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, NoMaximalReconvergenceMeldToMerge) {
  const std::string text = R"(
                              OpCapability Shader
                              OpCapability GroupNonUniformBallot
                              OpCapability GroupNonUniformArithmetic
                              OpMemoryModel Logical GLSL450
                              OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID %output
                              OpExecutionMode %main LocalSize 1 1 1
                              OpSource HLSL 660
                              OpName %type_RWStructuredBuffer_uint "type.RWStructuredBuffer.uint"
                              OpName %output "output"
                              OpName %main "main"
                              OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
                              OpDecorate %output DescriptorSet 0
                              OpDecorate %output Binding 0
                              OpDecorate %_runtimearr_uint ArrayStride 4
                              OpMemberDecorate %type_RWStructuredBuffer_uint 0 Offset 0
                              OpDecorate %type_RWStructuredBuffer_uint Block
                      %uint = OpTypeInt 32 0
                      %bool = OpTypeBool
                       %int = OpTypeInt 32 1
                     %int_0 = OpConstant %int 0
                     %int_1 = OpConstant %int 1
               %_runtimearr_uint = OpTypeRuntimeArray %uint
               %type_RWStructuredBuffer_uint = OpTypeStruct %_runtimearr_uint
               %_ptr_StorageBuffer_type_RWStructuredBuffer_uint = OpTypePointer StorageBuffer %type_RWStructuredBuffer_uint
                    %v3uint = OpTypeVector %uint 3
               %_ptr_Input_v3uint = OpTypePointer Input %v3uint
                      %void = OpTypeVoid
                        %15 = OpTypeFunction %void
                    %uint_3 = OpConstant %uint 3
               %_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
                    %output = OpVariable %_ptr_StorageBuffer_type_RWStructuredBuffer_uint StorageBuffer
               %gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
                      %main = OpFunction %void None %15
                        %18 = OpLabel
                        %19 = OpLoad %v3uint %gl_GlobalInvocationID
                              OpBranch %20
                        %20 = OpLabel
                              OpLoopMerge %21 %22 None
; CHECK:                      OpLoopMerge [[merge:%\w+]] [[continue:%\w+]]
                              OpBranch %23
                        %23 = OpLabel
                        %24 = OpCompositeExtract %uint %19 0
                        %25 = OpGroupNonUniformBroadcastFirst %uint %uint_3 %24
                        %26 = OpIEqual %bool %24 %25
                              OpSelectionMerge %27 None
                              OpBranchConditional %26 %28 %27
                        %28 = OpLabel
                        %29 = OpGroupNonUniformIAdd %int %uint_3 Reduce %int_1
                        %30 = OpBitcast %uint %29
                              OpBranch %21
; CHECK:          [[merge]] = OpLabel
; CHECK-NEXT:   [[t1:%\w+]] = OpGroupNonUniformIAdd %int %uint_3 Reduce %int_1
; CHECK-NEXT:   [[t2:%\w+]] = OpBitcast %uint [[t1]]
                        %27 = OpLabel
                              OpBranch %22
                        %22 = OpLabel
                              OpBranch %20
                        %21 = OpLabel
                        %31 = OpAccessChain %_ptr_StorageBuffer_uint %output %int_0 %24
                              OpStore %31 %30
                              OpReturn
                              OpFunctionEnd
)";

  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SinglePassRunAndMatch<BlockMergePass>(text, true);
}

TEST_F(BlockMergeTest, DontMergeLoopHeaderAndMergeBlock) {
  const std::string text = R"(OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
OpExecutionMode %main OriginUpperLeft
OpSource ESSL 310
OpName %main "main"
%void = OpTypeVoid
%3 = OpTypeFunction %void
%int = OpTypeInt 32 1
%int_1 = OpConstant %int 1
%int_n7 = OpConstant %int -7
%bool = OpTypeBool
%main = OpFunction %void None %3
%8 = OpLabel
OpBranch %9
%9 = OpLabel
%10 = OpPhi %int %int_1 %8 %int_n7 %11
%12 = OpSGreaterThan %bool %10 %int_n7
OpLoopMerge %13 %11 None
OpBranchConditional %12 %14 %13
%14 = OpLabel
OpBranch %15
%15 = OpLabel
OpLoopMerge %16 %17 None
OpBranch %16
%17 = OpLabel
OpBranch %15
%16 = OpLabel
OpBranch %11
%11 = OpLabel
OpBranch %9
%13 = OpLabel
OpReturn
OpFunctionEnd
)";

  SetTargetEnv(SPV_ENV_VULKAN_1_3);
  SinglePassRunAndCheck<BlockMergePass>(text, text, false);
}

// TODO(greg-lunarg): Add tests to verify handling of these cases:
//
//    More complex control flow
//    Others?

}  // namespace
}  // namespace opt
}  // namespace spvtools
