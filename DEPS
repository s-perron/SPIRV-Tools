use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'abseil_revision': '2424af2367325310e8e0caba7da3a022858346c4',

  'effcee_revision': '910ed15722d5d05c9d71ecf36c1a22243cb79b02',

  'googletest_revision': '3940de91897160fea4815998e08d0fa3c2fb077e',

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

