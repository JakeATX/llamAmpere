find_package(Threads REQUIRED)

llama_add_compile_flags()

# Optional args:
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build source)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs LINK)
    cmake_parse_arguments(LLAMA_BUILD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_BUILD_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_BUILD_LINK)
        set(LLAMA_BUILD_LINK llama-common)
    endif()

    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_BUILD_LINK})
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
endfunction()

function(llama_test target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    set(TEST_TARGET ${target})

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

function(llama_test_cmd target)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_NAME ${LLAMA_TEST_NAME})
    else()
        set(TEST_NAME ${target})
    endif()

    add_test(
        NAME ${TEST_NAME}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND ${target}
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_NAME} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# Builds and runs a test source file.
# Optional args:
# - NAME: name of the executable & test target (defaults to the source file name without extension)
# - LABEL: label for the test (defaults to main)
# - ARGS: arguments to pass to the test executable
# - WORKING_DIRECTORY
# - LINK: libraries to link against (defaults to llama-common, which also brings llama)
function(llama_build_and_test source)
    include(CMakeParseArguments)
    set(options)
    set(oneValueArgs NAME LABEL WORKING_DIRECTORY)
    set(multiValueArgs ARGS LINK)
    cmake_parse_arguments(LLAMA_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    set(TEST_SOURCES ${source} ${LLAMA_TEST_UNPARSED_ARGUMENTS})

    if (NOT LLAMA_TEST_LINK)
        set(LLAMA_TEST_LINK llama-common)
    endif()

    if (NOT DEFINED LLAMA_TEST_LABEL)
        set(LLAMA_TEST_LABEL "main")
    endif()
    if (NOT DEFINED LLAMA_TEST_WORKING_DIRECTORY)
        set(LLAMA_TEST_WORKING_DIRECTORY .)
    endif()
    if (DEFINED LLAMA_TEST_NAME)
        set(TEST_TARGET ${LLAMA_TEST_NAME})
    else()
        get_filename_component(TEST_TARGET ${source} NAME_WE)
    endif()

    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    target_compile_features(${TEST_TARGET} PRIVATE cxx_std_17)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS ${TEST_TARGET} RUNTIME)
    endif()
    target_link_libraries(${TEST_TARGET} PRIVATE ${LLAMA_TEST_LINK})

    add_test(
        NAME ${TEST_TARGET}
        WORKING_DIRECTORY ${LLAMA_TEST_WORKING_DIRECTORY}
        COMMAND $<TARGET_FILE:${TEST_TARGET}>
        ${LLAMA_TEST_ARGS})

    set_property(TEST ${TEST_TARGET} PROPERTY LABELS ${LLAMA_TEST_LABEL})
endfunction()

# build test-tokenizer-0 target once and add many tests
llama_build(test-tokenizer-0.cpp)

llama_test(test-tokenizer-0 NAME test-tokenizer-0-bert-bge        LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-bert-bge.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-command-r       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-command-r.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-coder  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-coder.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-deepseek-llm    LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-deepseek-llm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-falcon          LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gemma-4         LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gemma-4.gguf --check-decode)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-gpt-2           LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-bpe       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-llama-spm       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-mpt             LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-phi-3           LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-phi-3.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen2           LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen2.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-qwen35          LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-qwen35.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-refact          LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
llama_test(test-tokenizer-0 NAME test-tokenizer-0-starcoder       LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

if (NOT WIN32)
    llama_test_cmd(
        ${CMAKE_CURRENT_SOURCE_DIR}/test-tokenizers-repo.sh
        NAME test-tokenizers-ggml-vocabs
        LABEL behavior
        WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
        ARGS https://huggingface.co/ggml-org-dev/vocabs ${PROJECT_SOURCE_DIR}/models/ggml-vocabs
    )
endif()

if (LLAMA_LLGUIDANCE)
    llama_build_and_test(test-grammar-llguidance.cpp  LABEL "main behavior"  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf)
endif ()

llama_build(test-recurrent-state-rollback.cpp)
llama_build(test-gdn-replay-lanes.cpp) # [TAG_GDN_REPLAY_LANES] joint vs split replay; needs a GDN model (-m)
llama_build(test-dspine-rollback.cpp)
llama_build(test-fit-recurrent-noalloc.cpp)
llama_build(test-output-buffer-reuse.cpp)
llama_build(test-peer-experts.cpp LINK ggml) # GGML_PEER_EXPERTS transport; needs a CUDA device (same device twice is fine)
llama_build(test-inkmiss-cpu.cpp LINK ggml) # [INKMISS] CPU MUL_MAT_ID hashes: run flags off vs GGML_CPU_NVFP4_GROUPED=1 GGML_CPU_AI4_SCHEDULE=1, must match

llama_build_and_test(test-gdn-ingredient-replay.cpp LABEL "turbo")
llama_build_and_test(test-kq-mask-pack.cpp)
llama_build_and_test(test-gdn-prefix-lanes.cpp LABEL "turbo") # [TAG_GDN_REPLAY_LANES] per-lane prefix mask
llama_build_and_test(test-sjkvarn-codec.cpp LABEL "turbo")
target_include_directories(test-sjkvarn-codec PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
llama_build_and_test(test-sjkvarn-attn.cpp LABEL "turbo")
target_include_directories(test-sjkvarn-attn PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
llama_build_and_test(test-sjkvarn-fused-rot.cpp LABEL "turbo")
llama_build_and_test(test-sjkvarn-state.cpp LABEL "turbo")
target_include_directories(test-sjkvarn-state PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
llama_build_and_test(test-sjkvarn-split.cpp LABEL "turbo")
target_include_directories(test-sjkvarn-split PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
llama_build_and_test(test-sjkvarn-fill.cpp LABEL "turbo")
# LLAMA_LSA_RESIDENT=auto slot sizing (pure function, no model)
llama_build_and_test(test-lsa-resident-fit.cpp LABEL "turbo")
# FLASH_ATTN_EXT src[9]: upstream kv_rows vs the LSA LSE output (accessors)
llama_build_and_test(test-lsa-src9.cpp LABEL "turbo")
# [D1] exact head rescore: select, sidecar checks, scope, row states, threaded == serial (pure, no model)
llama_build_and_test(test-d1-rescore.cpp LABEL "turbo")
llama_build_and_test(test-head2.cpp) # [HEAD2] CPU reference of the exact two-stage head (optional args: model.gguf rows)
llama_build_and_test(test-moe-cache-policy.cpp) # [INKCACHEUX] expert-cache controls, heat file, miss split fit (pure, no model)
# prompt-cache state round trip over every KV / drafter / recurrent-state type (needs an MTP model: -m)
llama_build(test-prompt-cache-types.cpp)

# llama_context state tests: KV and recurrent state save, load, copy, rollback
# (upstream replaced test-save-load-state with these; the fork keeps its own
#  test-recurrent-state-rollback gates as well, registered after test-llama-context below)
set(LLAMA_CONTEXT_TEST_SRC
    test-llama-context.cpp
    llama-context/cases-kv.cpp
    llama-context/cases-rs.cpp
    llama-context/test-case.cpp
    llama-context/test-log.cpp
    llama-context/test-runner.cpp
    llama-context/test-util.cpp
)
llama_build(${LLAMA_CONTEXT_TEST_SRC})

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # these tests are disabled on Windows because they use internal functions not exported with LLAMA_API (when building with shared libraries)
    llama_build_and_test(test-unicode.cpp  LABEL behavior  LINK llama)
    if (NOT WIN32)
        llama_build_and_test(test-lsa-state.cpp LABEL "turbo")
        llama_build_and_test(test-lsa-multi.cpp LABEL "turbo")
    endif()
    llama_build_and_test(test-sampling.cpp  LABEL behavior)   # fork: uses common/sampling.h, so keep the llama-common default link
    llama_build_and_test(test-sampler-topk-exact.cpp)
    llama_build_and_test(test-sampler-topk-law.cpp)    # [HOSTGAP08]
    llama_build_and_test(test-spec-beat.cpp)
    llama_build_and_test(test-reasoning-budget.cpp  LABEL behavior)
    llama_build_and_test(test-grammar-parser.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-grammar-integration.cpp  LABEL behavior)
    llama_build_and_test(test-llama-grammar.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-grammar-clone.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-batch-alloc.cpp  LABEL behavior  LINK llama)
    llama_build_and_test(test-kv-cache.cpp LABEL behavior)
    target_include_directories(test-kv-cache PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-chat.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    target_include_directories(test-chat PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-chat PRIVATE server-context)
    llama_build_and_test(test-prompt-cache-disk.cpp)
    target_include_directories(test-prompt-cache-disk PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-prompt-cache-disk PRIVATE server-context)
    llama_build_and_test(test-loop-guard.cpp)
    target_include_directories(test-loop-guard PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-loop-guard PRIVATE server-context)
    llama_build_and_test(test-eval-cache.cpp)
    target_include_directories(test-eval-cache PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-eval-cache PRIVATE server-context)
    llama_build_and_test(test-server-component.cpp  LABEL behavior)
    target_include_directories(test-server-component PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
    target_link_libraries(test-server-component PRIVATE server-context)
    llama_build_and_test(test-json-schema-to-grammar.cpp  LABEL behavior)

    if (NOT GGML_BACKEND_DL)
        llama_build(test-quantize-stats.cpp)
    endif()

    llama_build(test-gbnf-validator.cpp LINK llama)

    # build test-tokenizer-1-bpe target once and add many tests
    llama_build(test-tokenizer-1-bpe.cpp)

    # TODO: disabled due to slowness
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-aquila     LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-aquila.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-falcon     LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-falcon.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-2      LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-2.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-gpt-neox   LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-gpt-neox.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-llama-bpe  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-bpe.gguf --ignore-merges)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-mpt        LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-mpt.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-refact     LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-refact.gguf)
    #llama_test(test-tokenizer-1-bpe NAME test-tokenizer-1-starcoder  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-starcoder.gguf)

    # build test-tokenizer-1-spm target once and add many tests
    llama_build(test-tokenizer-1-spm.cpp)

    llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-llama-spm  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-llama-spm.gguf)
    #llama_test(test-tokenizer-1-spm  NAME test-tokenizer-1-baichuan  LABEL behavior  ARGS ${PROJECT_SOURCE_DIR}/models/ggml-vocab-baichuan.gguf)

    # llama_build_and_test(test-double-float.cpp) # SLOW

    # fork: test-llama-archs, the generated models and test-fusion moved to top level with upstream (below)
    llama_build_and_test(test-mtp-state.cpp)
    llama_build(test-mtp-q5-head.cpp)
    llama_build(test-mtp-qsa-compact.cpp)
    target_include_directories(test-mtp-qsa-compact PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_test(test-mtp-state NAME test-prompt-tail-chunks ARGS --tail-chunks)
    llama_test(test-mtp-state NAME test-mtp-draft-ring-restore ARGS --draft-ring-restore-only)
endif()

llama_build(test-llama-archs.cpp)
if (GGML_CPU AND (NOT WIN32 OR NOT BUILD_SHARED_LIBS))
    llama_build(test-fp8.cpp)
    set(FP8_MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models-fp8")
    file(MAKE_DIRECTORY "${FP8_MODEL_DIR}")
    llama_test(test-llama-archs NAME test-generate-fp8-models LABEL "main behavior backend"
        ARGS -a llama -s 1 -o "${FP8_MODEL_DIR}")
    set_tests_properties(test-generate-fp8-models PROPERTIES FIXTURES_SETUP fp8-models)
    llama_test(test-fp8 NAME test-fp8-requantize LABEL "main behavior backend"
        ARGS --models "${FP8_MODEL_DIR}")
    llama_test(test-fp8 NAME test-fp8-kv LABEL "main behavior backend"
        ARGS --models "${FP8_MODEL_DIR}" --kv)
    set_tests_properties(test-fp8-requantize test-fp8-kv PROPERTIES FIXTURES_REQUIRED fp8-models)
endif()

set(MODEL_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-models/")
file(MAKE_DIRECTORY "${MODEL_DIR}")

llama_test(
    test-llama-archs
    NAME test-generate-models
    LABEL "main behavior backend"
    ARGS -o "${MODEL_DIR}"
)
set_tests_properties(test-generate-models PROPERTIES
    FIXTURES_SETUP generate-models
)

# Target model for test-speculative-rejection-sampling
set(SPEC_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}/test-speculative-models/target")
file(MAKE_DIRECTORY "${SPEC_TARGET_DIR}")
llama_test(
    test-llama-archs
    NAME test-generate-speculative-target
    LABEL main
    ARGS -a llama -s 999 -o "${SPEC_TARGET_DIR}"
)
set_tests_properties(
    test-generate-speculative-target
    PROPERTIES FIXTURES_SETUP speculative-model
)

# To avoid recreating the model use the fixture exclude any option:
# -FA "^speculative-model$"
llama_build_and_test(test-speculative-rejection-sampling.cpp ARGS
    -m "${SPEC_TARGET_DIR}/llama-dense.gguf")
set_tests_properties(test-speculative-rejection-sampling PROPERTIES FIXTURES_REQUIRED speculative-model)

# Test context state handling across all architectures, using the generated dummy models
llama_test(
    test-llama-context
    LABEL "main behavior backend"
    ARGS --models "${MODEL_DIR}"
)
# abort on graph reallocations that the initial reserve should have covered
set_tests_properties(test-llama-context PROPERTIES
    FIXTURES_REQUIRED generate-models
    ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1"
)

# fork: recurrent-state rollback and -fit probe gates on the generated dummy models
# (they use internal symbols, so the same Windows shared-lib guard as before applies)
if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    llama_test(
        test-recurrent-state-rollback
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-gdn-replay-lanes
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-gdn-replay-lanes PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-gdn-replay-lanes
        NAME test-gdn-replay-lanes-short
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-gdn-replay-lanes-short PROPERTIES
        FIXTURES_REQUIRED generate-models
        ENVIRONMENT "LLAMA_GDN_REPLAY_SHORT=1"
    )

    # [DSPINE-V07] 16-row verify rounds rolled back to every depth, ring and gdn_replay layouts
    llama_test(
        test-dspine-rollback
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-dspine-rollback PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    # the -fit probe (no_alloc) must size the recurrent state without allocating it
    llama_test(
        test-fit-recurrent-noalloc
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35-dense.gguf"
    )
    set_tests_properties(test-fit-recurrent-noalloc PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    # same check for the DSV4 compressor states
    llama_test(
        test-fit-recurrent-noalloc
        NAME test-fit-recurrent-noalloc-dsv4
        LABEL main
        ARGS -m "${MODEL_DIR}/deepseek4-moe.gguf"
    )
    set_tests_properties(test-fit-recurrent-noalloc-dsv4 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-nemotron-h
        LABEL main
        ARGS -m "${MODEL_DIR}/nemotron_h-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-nemotron-h PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-dsv4
        LABEL main
        ARGS -m "${MODEL_DIR}/deepseek4-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-dsv4 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-kimi-k3
        LABEL main
        ARGS -m "${MODEL_DIR}/kimi-k3-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-kimi-k3 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-lfm2
        LABEL main
        ARGS -m "${MODEL_DIR}/lfm2-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-lfm2 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-lfm2moe
        LABEL main
        ARGS -m "${MODEL_DIR}/lfm2moe-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-lfm2moe PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-bailingmoe3
        LABEL main
        ARGS -m "${MODEL_DIR}/bailingmoe3-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-bailingmoe3 PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-qwen4exp
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-qwen4exp PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-qwen35moe
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen35moe-moe.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-qwen35moe PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-nemotron-h-moe
        LABEL main
        ARGS -m "${MODEL_DIR}/nemotron_h_moe-dense.gguf"
    )
    set_tests_properties(test-recurrent-state-rollback-nemotron-h-moe PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
endif()

llama_build(test-fusion.cpp)

llama_build_and_test(test-chat-peg-parser.cpp peg-parser/simple-tokenize.cpp  LABEL behavior)
llama_build_and_test(test-jinja.cpp  LABEL behavior)
llama_test(test-jinja  NAME test-jinja-py  ARGS -py  LABEL "python behavior")
llama_test_cmd(python3
    NAME test-conversion-fp8
    LABEL "python behavior"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    ARGS -m unittest discover -s tests -p test_conversion_fp8.py -v
)
llama_test_cmd(python3
    NAME test-kv-cache-scales
    LABEL "python behavior"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    ARGS -m unittest discover -s tests -p test_kv_cache_scales.py -v
)
llama_build_and_test(test-json-schema.cpp  LABEL behavior)
llama_build_and_test(test-chat-auto-parser.cpp  LABEL behavior  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
llama_build_and_test(test-chat-template.cpp  LABEL behavior)
# debug tool for chat template differential analysis (not registered as a test, run it manually)
llama_build(test-chat-analysis.cpp)
llama_build_and_test(test-log.cpp  LABEL behavior)
llama_build_and_test(
    test-peg-parser.cpp
    peg-parser/simple-tokenize.cpp
    peg-parser/test-basic.cpp
    peg-parser/test-gbnf-generation.cpp
    peg-parser/test-json-parser.cpp
    peg-parser/test-python-dict-parser.cpp
    peg-parser/test-unicode.cpp
    peg-parser/tests.h
    LABEL behavior
)

if (NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "s390x")
    set(MODEL_NAME "stories15M-q4_0.gguf")
    set(MODEL_HASH "SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739")
else()
    set(MODEL_NAME "stories15M-be.Q4_0.gguf")
    set(MODEL_HASH "SHA256=9aec857937849d976f30397e97eb1cabb53eb9dcb1ce4611ba8247fb5f44c65d")
endif()
set(MODEL_DEST "${CMAKE_BINARY_DIR}/${MODEL_NAME}")

add_test(NAME test-download-model COMMAND ${CMAKE_COMMAND}
    -DREPO=ggml-org-dev/stories15M
    -DDEST=${MODEL_DEST}
    -DNAME=${MODEL_NAME}
    -DHASH=${MODEL_HASH}
    -P ${PROJECT_SOURCE_DIR}/cmake/download-models.cmake
)
set_tests_properties(test-download-model PROPERTIES FIXTURES_SETUP test-download-model)

# the context state cases against a real model, the sweep above uses the generated dummy ones
llama_test(test-llama-context  NAME test-llama-context-model  LABEL "model behavior backend"  ARGS -m "${MODEL_DEST}")
set_tests_properties(test-llama-context-model PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-thread-safety.cpp  LABEL "main backend"  ARGS -m "${MODEL_DEST}" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 -np 4 -t 2)
set_tests_properties(test-thread-safety PROPERTIES FIXTURES_REQUIRED test-download-model)

llama_build_and_test(test-arg-parser.cpp  LABEL behavior)
if (NOT WIN32)
    # UPSYNC 0.7: local FP8/TURBO2 type-ID load guard + FLASH_ATTN_EXT kv_rows (src[9]) / SJ-KVaRN slot guards (uses fork)
    llama_build_and_test(test-upsync-guards.cpp  LABEL behavior)
    # UPSYNC 0.7: KV state save/load with block KV streaming on (CPU: refused at creation; GPU part exits 77 without CUDA)
    llama_build_and_test(test-kv-stream-state.cpp  LABEL behavior)
    set_tests_properties(test-kv-stream-state PROPERTIES SKIP_RETURN_CODE 77)
endif()
if (NOT BUILD_SHARED_LIBS)
    llama_build_and_test(test-kv-stream-plan.cpp ${PROJECT_SOURCE_DIR}/src/llama-kv-stream-plan.cpp)
    target_include_directories(test-kv-stream-plan PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_build_and_test(test-kv-stream-config.cpp ${PROJECT_SOURCE_DIR}/src/llama-kv-stream-config.cpp)
    target_include_directories(test-kv-stream-config PRIVATE ${PROJECT_SOURCE_DIR}/src)
endif()
llama_build_and_test(test-kv-stream-softmax.cpp llama-kv-stream-softmax.cpp)
if (GGML_CUDA)
    enable_language(CUDA)
    add_executable(test-moe-hf-copy test-moe-hf-copy.cu)
    set_target_properties(test-moe-hf-copy PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES)
    if (LLAMA_TESTS_INSTALL)
        install(TARGETS test-moe-hf-copy RUNTIME)
    endif()
    llama_test(test-moe-hf-copy LABEL cuda)
    set_tests_properties(test-moe-hf-copy PROPERTIES SKIP_RETURN_CODE 77)
    llama_build_and_test(test-kv-stream-cuda-buffer.cpp LABEL cuda)
    target_include_directories(test-kv-stream-cuda-buffer PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
    # The test exits 77 when no CUDA device is present (e.g. a CUDA build run
    # on a GPU-less CI runner). ctest only treats that as a skip when
    # SKIP_RETURN_CODE says so; without it the skip is recorded as a failure.
    set_tests_properties(test-kv-stream-cuda-buffer PROPERTIES SKIP_RETURN_CODE 77)
    llama_build_and_test(test-kv-stream-cuda-attn.cpp LABEL cuda)
    target_include_directories(test-kv-stream-cuda-attn PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
    llama_build_and_test(test-kv-stream-cuda-set-rows.cpp LABEL cuda)
    target_include_directories(test-kv-stream-cuda-set-rows PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
    # [PP27] AIF25_U exact prompt kernel vs the 8-token kernel, bit for bit (exits 77 without a CUDA device)
    llama_build_and_test(test-aif25-mmq.cpp LABEL cuda)
    target_include_directories(test-aif25-mmq PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    set_tests_properties(test-aif25-mmq PROPERTIES SKIP_RETURN_CODE 77)
    # the kernel variant is read once per process: one ctest entry per built tile/strips variant
    foreach(v 32_1 128_1 32_2 64_2)
        string(REPLACE "_" ";" vp ${v})
        list(GET vp 0 v_tile)
        list(GET vp 1 v_strips)
        add_test(NAME test-aif25-mmq-${v} COMMAND $<TARGET_FILE:test-aif25-mmq>)
        set_tests_properties(test-aif25-mmq-${v} PROPERTIES
            ENVIRONMENT "GGML_CUDA_AIF25_MMQ=${v_tile};GGML_CUDA_AIF25_MMQ_STRIPS=${v_strips}"
            SKIP_RETURN_CODE 77 LABELS cuda)
    endforeach()
    # [KAIMULTI] prompt kernel only from 64 tokens: widths 9..32 take the decode-once narrow kernel, 33..63 stay on mmv
    add_test(NAME test-aif25-mmq-narrow COMMAND $<TARGET_FILE:test-aif25-mmq>)
    set_tests_properties(test-aif25-mmq-narrow PROPERTIES
        ENVIRONMENT "GGML_CUDA_AIF25_MMQ_MIN=64" SKIP_RETURN_CODE 77 LABELS cuda)
    # [OPT10] cp.async fractional AI+ matvec (GGML_CUDA_AIF_ASYNC=1): verify widths 9..32 on the async narrow tile and
    # 33..63 on the async mmv tiling; with the narrow kernel off, 9..63 all take the async mmv tiling
    add_test(NAME test-aif25-mmq-async COMMAND $<TARGET_FILE:test-aif25-mmq>)
    set_tests_properties(test-aif25-mmq-async PROPERTIES
        ENVIRONMENT "GGML_CUDA_AIF25_MMQ_MIN=64;GGML_CUDA_AIF_ASYNC=1" SKIP_RETURN_CODE 77 LABELS cuda)
    add_test(NAME test-aif25-mmq-async-nb1 COMMAND $<TARGET_FILE:test-aif25-mmq>)
    set_tests_properties(test-aif25-mmq-async-nb1 PROPERTIES
        ENVIRONMENT "GGML_CUDA_AIF25_MMQ_MIN=64;GGML_CUDA_AIF_NARROW=0;GGML_CUDA_AIF_ASYNC=1" SKIP_RETURN_CODE 77 LABELS cuda)
    # [KAIFUSE] shared AI+ prep (R1) and the AIF draft shortlist rows (R2), bit for bit (exits 77 without a CUDA device)
    llama_build_and_test(test-kaifuse.cpp LABEL cuda)
    set_tests_properties(test-kaifuse PROPERTIES SKIP_RETURN_CODE 77)
    llama_build(test-kaifuse-logits.cpp) # [KAIFUSE] real-model logit hashes for the R1 identity check (needs a model)
endif()
# [OPT10] CPU reference of the cp.async fractional AI+ matvec addressing and int -> float conversion (no GPU needed)
llama_build_and_test(test-aif-async-addr.cpp)
llama_build_and_test(test-model-resolution.cpp  LABEL behavior)
# the test serves its repos from an httplib server, and the library links it privately
target_link_libraries(test-model-resolution PRIVATE cpp-httplib)
llama_build_and_test(test-speculative-adaptive.cpp)
llama_build_and_test(test-speculative-copy.cpp)
llama_build_and_test(test-speculative-copy-ctl.cpp)
# [DSPINE-V07] lattice walk, verify-length planner, p/q exactness at T=1 (pure CPU, no model)
llama_build_and_test(test-spec-lattice.cpp)

if (NOT LLAMA_SANITIZE_ADDRESS AND NOT GGML_SCHED_NO_REALLOC)
  # TODO: repair known memory leaks
  llama_build_and_test(test-opt.cpp  LABEL "main behavior backend"  LINK ggml)
endif()
if (GGML_METAL AND NOT GGML_BACKEND_DL)
    # calls the Metal backend's graph optimizer and fusion table directly
    llama_build_and_test(test-metal-graph-optimize.cpp  LABEL "main behavior backend"  LINK ggml ggml-metal)
    target_include_directories(test-metal-graph-optimize PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src ${PROJECT_SOURCE_DIR}/ggml/src/ggml-metal)
endif()

llama_build_and_test(test-backend-ops.cpp LINK ggml Threads::Threads)
llama_build_and_test(test-fattn-prefill-policy.cpp)
llama_build(test-i8qk-flagcache.cpp)
llama_build_and_test(test-fa-skip-dead.cpp) # [ROUNDFIX] dead-tile skip bit-identity; skips without a GPU backend
llama_build_and_test(test-topk-select-fast.cpp) # [ROUNDFIX] fast radix select bit-identity; skips without a GPU backend
llama_build_and_test(test-lid-gather.cpp) # [ROUNDFIX] pooled-key gather read in place, bit-identity; skips without a GPU backend
llama_build_and_test(test-mtp-vocab.cpp)
llama_build_and_test(test-mtp-vocab-builtin.cpp ../src/llama-mtp-vocab-builtin.cpp)
llama_build_and_test(test-gdn-replay-split.cpp)
# [TAG_GDN_REPLAY_LANES] short static rounds: joined (SHORT=1) and split (SHORT=1 SHORT_JOIN=0)
llama_test(test-gdn-replay-split NAME test-gdn-replay-split-short)
set_tests_properties(test-gdn-replay-split-short PROPERTIES ENVIRONMENT "LLAMA_GDN_REPLAY_SHORT=1")
llama_test(test-gdn-replay-split NAME test-gdn-replay-split-short-nojoin)
set_tests_properties(test-gdn-replay-split-short-nojoin PROPERTIES ENVIRONMENT "LLAMA_GDN_REPLAY_SHORT=1;LLAMA_GDN_REPLAY_SHORT_JOIN=0")
llama_build_and_test(test-add-rms-q8.cpp) # [#46] skips without a GPU backend
llama_build_and_test(test-hc-norm-fuse.cpp) # [HCPN] skips without a GPU backend
llama_build_and_test(test-exl3-ffn-bridge.cpp) # [#74] skips without a GPU backend
llama_build_and_test(test-exl3-weight-major.cpp) # [#73] skips without a GPU backend
llama_build_and_test(test-exl3-decode.cpp) # [#41] skips without a GPU backend
llama_build_and_test(test-exl3-glue-fold.cpp) # [#136] skips without a GPU backend
if (NOT GGML_BACKEND_DL AND GGML_CPU)
    add_executable(test-exl3-cpu test-exl3-cpu.cpp)
    target_link_libraries(test-exl3-cpu PRIVATE ggml ggml-cpu)
    llama_test(test-exl3-cpu)
    target_include_directories(test-exl3-cpu PRIVATE ../ggml/src ../ggml/src/ggml-cpu)
    set_tests_properties(test-exl3-cpu PROPERTIES SKIP_RETURN_CODE 77)
endif()
add_executable(test-fattn-mma-config test-fattn-mma-config.cpp)
target_include_directories(test-fattn-mma-config PRIVATE ${PROJECT_SOURCE_DIR}/ggml/include)
llama_test(test-fattn-mma-config)

add_executable(test-rpb-knobs test-rpb-knobs.cpp)
target_include_directories(test-rpb-knobs PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda)
llama_test(test-rpb-knobs)

add_executable(test-dspark-head test-dspark-head.cpp)
target_include_directories(test-dspark-head PRIVATE ${PROJECT_SOURCE_DIR}/src)
llama_test(test-dspark-head)

llama_build_and_test(test-fattn-alloc-route.cpp) # [#64] skips without a GPU backend
llama_build_and_test(test-fattn-bounded-prefill.cpp) # [#22/#29] forks one child per GGML_CUDA_PREFILL_KV_MIB arm; skips without a GPU backend
llama_build_and_test(test-mtp-chain-sample.cpp) # [#69] CPU always, the GPU when present
llama_build_and_test(test-gdn-ingr-compact.cpp) # [#63] CPU always, the GPU when present
llama_build_and_test(test-draft-vocab-hot.cpp)
llama_build_and_test(test-draft-vocab-compact.cpp)
llama_build_and_test(test-nmse-masked-inf.cpp)
if (NOT LLAMA_USE_SYSTEM_GGML)
    llama_build(test-moe-cache.cpp)
    if (GGML_CUDA)
        # The test compares the cache's budget grants against cudaMemGetInfo,
        # the same query the cache uses. On UMA devices (GB10, Jetson) the
        # backend device memory API reports /proc/meminfo instead, so the
        # reserve calculation must match the cache's own view.
        find_package(CUDAToolkit QUIET)
        if (CUDAToolkit_FOUND)
            target_compile_definitions(test-moe-cache PRIVATE GGML_CUDA=1)
            target_include_directories(test-moe-cache PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
            target_link_libraries(test-moe-cache PRIVATE CUDA::cudart)
        endif()
    endif()
    llama_test(test-moe-cache LABEL "main;backend")
    # The test exits 77 when no CUDA device is present. ctest only treats that
    # as a skip when SKIP_RETURN_CODE says so; without it, every CPU-only
    # runner records the skip as a failure.
    set_tests_properties(test-moe-cache PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)
endif()

# Manual fixed-token model gate; needs model paths and input IDs supplied by the caller.
llama_build(test-moe-prefetch-model.cpp)
llama_build(test-kq-mask-host.cpp)   # host-KV KQ mask: -ngl 0 logits finite and equal to LLAMA_KQ_MASK_DEVICE=0 (gate runs it twice)

llama_build_and_test(test-moe-cache-fit.cpp)
llama_build_and_test(test-moe-r2-plan.cpp)   # route 2: card slots sized by real AI+ bytes
llama_build_and_test(test-moe-expert-keep.cpp) # [#215/#217] CPU-only expert keep policy
llama_build_and_test(test-decode-phase.cpp)     # one-shot decode tag gates the #217 trunk keep map
llama_build_and_test(test-fit-extra.cpp)
llama_build_and_test(test-moe-bank.cpp)
llama_build_and_test(test-moe-ai-rows.cpp)   # fork AIPFIX D2: host AI+ slab kernel vs reference
llama_build_and_test(test-ai4-avx2.cpp test-ai4-avx2-levels.cpp)   # opt-in GGML_CPU_AI4_AVX2=1 AI+ unit CPU path vs the reference path
if (NOT MSVC AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64|i[3-6]86")
    # only this helper TU; the test calls it after a runtime AVX2 + FMA check
    set_source_files_properties(test-ai4-avx2-levels.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()
llama_build_and_test(test-moe-iq-signs.cpp)   # compile-time IQ sign bytes vs runtime ksigns_iq2xs
llama_build_and_test(test-moe-cs-plan.cpp)   # fork AIPFIX: CPU-split planner greedy step + per-type probe cost table

llama_build_and_test(test-backend-tensor.cpp LINK ggml)
if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    llama_build_and_test(test-model-mmap.cpp)
endif()

if (BUILD_SHARED_LIBS)
    llama_build_and_test(test-llama-mmap.cpp ${PROJECT_SOURCE_DIR}/src/llama-impl.cpp ${PROJECT_SOURCE_DIR}/src/llama-mmap.cpp LABEL behavior)
    target_compile_definitions(test-llama-mmap PRIVATE LLAMA_BUILD)
else()
    llama_build_and_test(test-llama-mmap.cpp LABEL behavior)
endif()

llama_build_and_test(test-tiled-mulmat.cpp  LABEL "main behavior backend"  LINK ggml)
llama_build_and_test(test-model-load-cancel.cpp LABEL "model behavior")
llama_build_and_test(test-autorelease.cpp       LABEL "model behavior")
llama_build_and_test(test-backend-sampler.cpp   LABEL "model behavior backend")

if (APPLE)
    llama_build(test-rset-release.cpp)
    llama_build_and_test(test-host-buffer-release.cpp LINK ggml)
endif()

if (NOT GGML_BACKEND_DL AND GGML_CPU)
    # these tests use the backends directly and cannot be built with dynamic loading
    llama_build_and_test(test-barrier.cpp  LABEL "main behavior"  LINK ggml)
    llama_build_and_test(test-quantize-fns.cpp  LABEL "main behavior"  LINK ggml)
    llama_build_and_test(test-quantize-perf.cpp  LABEL "main behavior backend"  LINK ggml)
    target_include_directories(test-quantize-fns PRIVATE ../ggml/src)
    target_include_directories(test-quantize-perf PRIVATE ../ggml/src)
    llama_build_and_test(test-rope.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-col2im-1d.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-conv-1d.cpp  LABEL "main behavior backend"  LINK ggml)
    llama_build_and_test(test-gelu-quick.cpp  LABEL "main behavior backend"  LINK ggml)
endif()

# fork: test the actual host dispatch against direct kernels; the reference build has explicit ISA flags.
if (NOT GGML_BACKEND_DL AND GGML_CPU AND TARGET ggml-cuda AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" AND NOT MSVC)
    foreach(isa avx2 scalar)
        set(LLAMA_TEST_NAME test-moe-cs-dispatch-${isa})
        llama_build(test-moe-ai-rows.cpp)
        target_compile_definitions(${LLAMA_TEST_NAME} PRIVATE MOE_CS_TEST_DISPATCH)
        target_link_libraries(${LLAMA_TEST_NAME} PRIVATE ggml-cuda ggml-cpu)
        target_compile_options(${LLAMA_TEST_NAME} PRIVATE -march=x86-64)
        if (isa STREQUAL "avx2")
            target_compile_options(${LLAMA_TEST_NAME} PRIVATE -mavx2 -mfma -mf16c -mbmi2)
        endif()
    endforeach()
    unset(LLAMA_TEST_NAME)
endif()

# fork MOEGAP: host expert pool bandwidth bench (not a ctest; run by hand on the target box)
if (NOT GGML_BACKEND_DL AND GGML_CPU AND TARGET ggml-cuda AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    llama_build(test-moe-cs-bw.cpp)
    target_link_libraries(test-moe-cs-bw PRIVATE ggml-cuda ggml-cpu)
    llama_build(test-moe-iq-rows.cpp)
    target_link_libraries(test-moe-iq-rows PRIVATE ggml-cuda ggml-cpu)
endif()

# libmtmd
llama_build_and_test(test-mtmd-c-api.c  LABEL behavior  LINK mtmd)

llama_build_and_test(test-mtmd-impl.cpp  LABEL behavior  LINK mtmd)

# GGUF model data fetcher library for tests that need real model metadata
# Only compile when cpp-httplib has SSL support (CPPHTTPLIB_OPENSSL_SUPPORT)
if (TARGET cpp-httplib)
    get_target_property(_cpp_httplib_defs cpp-httplib INTERFACE_COMPILE_DEFINITIONS)
    if (_cpp_httplib_defs MATCHES "CPPHTTPLIB_OPENSSL_SUPPORT")
        add_library(gguf-model-data STATIC gguf-model-data.cpp)
        target_link_libraries(gguf-model-data PRIVATE llama-common cpp-httplib)
        target_include_directories(gguf-model-data PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

        add_executable(test-gguf-model-data test-gguf-model-data.cpp)
        target_link_libraries(test-gguf-model-data PRIVATE gguf-model-data llama-common)
        llama_test(test-gguf-model-data  LABEL "model behavior")

        # test-quant-type-selection requires gguf-model-data for remote model metadata
        llama_build_and_test(test-quant-type-selection.cpp  LABEL "model behavior"  LINK llama)
        target_link_libraries(test-quant-type-selection PRIVATE gguf-model-data)
        target_compile_definitions(test-quant-type-selection PRIVATE
            SNAPSHOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/snapshots")
    endif()
endif()

# dummy executable - not installed
get_filename_component(TEST_TARGET test-c.c NAME_WE)
add_executable(${TEST_TARGET} test-c.c)
target_link_libraries(${TEST_TARGET} PRIVATE llama)

# TurboQuant WHT round-trip test (links ggml turbo codec symbols via llama)
llama_build_and_test(test-turbo-quant.c LABEL "turbo")
llama_build_and_test(test-drama3-rope.cpp)

if (NOT LLAMA_USE_SYSTEM_GGML)
    # Needs non-public ggml-impl.h
    llama_build_and_test(test-gguf.cpp  LABEL behavior  LINK ggml)

    # Needs non-public ggml{,-backend}-impl.h
    llama_build_and_test(test-alloc.cpp  LABEL "behavior backend"  LINK ggml)
endif()

llama_build(test-export-graph-ops.cpp)
if (TARGET gguf-model-data)
    target_link_libraries(test-export-graph-ops PRIVATE gguf-model-data)
    target_compile_definitions(test-export-graph-ops PRIVATE LLAMA_HF_FETCH)
endif()

if (NOT LLAMA_USE_SYSTEM_GGML)
    add_executable(test-ple-cache test-ple-cache.cpp ../src/llama-ple-cache.cpp ../src/llama-moe-arena.cpp)
    target_link_libraries(test-ple-cache PRIVATE llama ggml ggml-cpu ggml-base)
    llama_test(test-ple-cache LABEL "main;cpu")

    add_executable(test-moe-arena test-moe-arena.cpp ../src/llama-moe-arena.cpp)
    target_link_libraries(test-moe-arena PRIVATE llama ggml ggml-cpu ggml-base)
    llama_test(test-moe-arena LABEL "main;cpu")
    llama_test(test-moe-arena NAME test-moe-arena-thp ARGS --thp LABEL "main;cpu")
    llama_test(test-moe-arena NAME test-moe-arena-ram ARGS --ram LABEL "main;cpu")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT LLAMA_USE_SYSTEM_GGML)
    add_executable(test-mmap-prefetch test-mmap-prefetch.cpp ../src/llama-mmap.cpp ../src/llama-ple-prefetch.cpp)
    target_link_libraries(test-mmap-prefetch PRIVATE llama ggml-cpu ggml-base)
    target_link_options(test-mmap-prefetch PRIVATE "LINKER:--wrap=mmap" "LINKER:--wrap=posix_madvise" "LINKER:--wrap=madvise" "LINKER:--wrap=pread" "LINKER:--wrap=posix_fadvise" "LINKER:--wrap=mincore" "LINKER:--wrap=_ZN15llama_moe_arena16available_memoryEv")
    target_include_directories(test-mmap-prefetch PRIVATE ${PROJECT_SOURCE_DIR}/src)
    llama_test(test-mmap-prefetch LABEL "main;cpu")
endif()

# fork (R4, NEXTPF): next-layer expert prefetch worker and bookkeeping, CPU only
add_executable(test-moe-prefetch test-moe-prefetch.cpp ../src/llama-moe-prefetch.cpp)
target_link_libraries(test-moe-prefetch PRIVATE ggml-base Threads::Threads)
llama_test(test-moe-prefetch LABEL "main;cpu")

if (UNIX AND NOT LLAMA_USE_SYSTEM_GGML)
    # safety: --load-mode dio reads through a plain bounce buffer and gives the same bytes as mmap
    add_executable(test-dio-bounce test-dio-bounce.cpp)
    target_link_libraries(test-dio-bounce PRIVATE llama ggml-base)
    llama_test(test-dio-bounce LABEL "main;cpu")
    llama_test(test-dio-bounce NAME test-aligned-alloc LABEL "main;cpu" ARGS --alloc-only)
    set_tests_properties(test-dio-bounce PROPERTIES SKIP_RETURN_CODE 77)
endif()

if (UNIX)
    add_executable(test-moe-generation-profile test-moe-generation-profile.cpp)
    llama_test(test-moe-generation-profile LABEL "main")
endif()

llama_build_and_test(test-gdn-conv-norm.cpp)
set_tests_properties(test-gdn-conv-norm PROPERTIES SKIP_RETURN_CODE 77)
llama_build_and_test(test-ait-format.cpp)
llama_build_and_test(test-ai4-u2.cpp)
llama_build_and_test(test-ai4-turbo-ids-route.cpp)   # [TURBO-IDS] direct-ids routing math vs the ids_helper contract (CPU only)
llama_build_and_test(test-gguf-deprecated-ai.cpp)

if (GGML_CUDA AND NOT GGML_BACKEND_DL) # test-magic2-v2 links ggml-cuda directly (a MODULE under GGML_BACKEND_DL)
    enable_language(CUDA)
    add_executable(test-magic2-v2 test-magic2-v2.cu)
    target_link_libraries(test-magic2-v2 PRIVATE ggml-base ggml-cuda)
    set_target_properties(test-magic2-v2 PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES 86)
    llama_test(test-magic2-v2 LABEL "cuda")
    set_tests_properties(test-magic2-v2 PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(bench-prefill-glue bench-prefill-glue.cu)
    target_link_libraries(bench-prefill-glue PRIVATE ggml-base ggml-cuda)
    target_include_directories(bench-prefill-glue PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda ${PROJECT_SOURCE_DIR}/ggml/src)
    set_target_properties(bench-prefill-glue PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES 86)
endif()

# Pure-CPU verification harness for prefill-glue numerics (zero CUDA runtime / GPU device usage)
add_executable(test-prefill-glue-cpu test-prefill-glue-cpu.cpp)
target_link_libraries(test-prefill-glue-cpu PRIVATE ggml-base)
target_include_directories(test-prefill-glue-cpu PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src/ggml-cuda ${PROJECT_SOURCE_DIR}/ggml/src)
llama_test(test-prefill-glue-cpu LABEL "main;cpu")
