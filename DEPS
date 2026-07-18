use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '1e6d60b2ca9356542fe62b73ab010424aa2796cf',

  'effcee_revision': '910ed15722d5d05c9d71ecf36c1a22243cb79b02',

  'googletest_revision': 'fa005b296f90faec4f352d7ab382287bf6548c8d',

  # Use a recent protobuf, which can depend on abseil
  'protobuf_revision': '35cd01f9fe9afbeea38cc7b979a3b6bfcde82c03',

  're2_revision': '972a15cedd008d846f1a39b2e88ce48d7f166cbd',

  'spirv_headers_revision': '29981f65241605e08b0ede4cfeb999fe3b723c6a',

  'mimalloc_revision': '76d3f8a934f9761e4ee75fa8b071e58d482f2758',
}

deps = {
  'external/abseil_cpp':
      Var('github') + '/abseil/abseil-cpp.git@' + Var('abseil_revision'),

  'external/effcee':
      Var('github') + '/google/effcee.git@' + Var('effcee_revision'),

  'external/googletest':
      Var('github') + '/google/googletest.git@' + Var('googletest_revision'),

  'external/protobuf':
      Var('github') + '/protocolbuffers/protobuf.git@' + Var('protobuf_revision'),

  'external/re2':
      Var('github') + '/google/re2.git@' + Var('re2_revision'),

  'external/spirv-headers':
      Var('github') +  '/KhronosGroup/SPIRV-Headers.git@' +
          Var('spirv_headers_revision'),

  'external/mimalloc':
      Var('github') + '/microsoft/mimalloc.git@' + Var('mimalloc_revision'),
}

