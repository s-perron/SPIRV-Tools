use_relative_paths = True

vars = {
  'github': 'https://github.com',

  'effcee_revision': 'c7b4db79f340f7a9981e8a484f6d5785e24242d1',

  # Pin to the last version of googletest that supports C++11.
  # Anything later requires C++14
  'googletest_revision': 'v1.12.0',

  # Use protobufs before they gained the dependency on abseil
  'protobuf_revision': 'v3.13.0.1',

  're2_revision': '954656f47fe8fb505d4818da1e128417a79ea500',
  'spirv_headers_revision': '36c7694279dfb3ea7e6e086e368bb8bee076792a',
}

deps = {
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
}

