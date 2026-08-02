use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': 'c27ad56a09cf6c95412020422e53c052f616addb',

  'effcee_revision': '910ed15722d5d05c9d71ecf36c1a22243cb79b02',

  'googletest_revision': '1b6f64d659944658a4c685b7bd9f04c1c3b8a39b',

  # Use a recent protobuf, which can depend on abseil
  'protobuf_revision': '35cd01f9fe9afbeea38cc7b979a3b6bfcde82c03',

  're2_revision': '972a15cedd008d846f1a39b2e88ce48d7f166cbd',

  'spirv_headers_revision': '4015a331f5ffd6fc5c6fa7b03e08fb4a692491d7',

  'mimalloc_revision': 'a3ca0e5e2eb283c2c9275f30872e30252a91b66c',
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

