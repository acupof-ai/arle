#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

struct ArleXGrammarCompiler;
struct ArleXGrammarCompiledGrammar;
struct ArleXGrammarMatcher;

const char* arle_xgrammar_version();
void arle_xgrammar_free_error(char* message);
int32_t arle_xgrammar_bitmask_size(int32_t vocab_size);

int arle_xgrammar_compiler_new(
    const char* const* encoded_vocab,
    std::size_t encoded_vocab_len,
    int32_t vocab_type,
    int32_t vocab_size,
    const int32_t* stop_token_ids,
    std::size_t stop_token_ids_len,
    uint8_t add_prefix_space,
    int32_t max_threads,
    uint8_t cache_enabled,
    int64_t max_memory_bytes,
    ArleXGrammarCompiler** out,
    char** error
);
void arle_xgrammar_compiler_free(ArleXGrammarCompiler* compiler);

int arle_xgrammar_compile_json_schema(
    ArleXGrammarCompiler* compiler,
    const char* schema,
    uint8_t strict_mode,
    ArleXGrammarCompiledGrammar** out,
    char** error
);
int arle_xgrammar_compile_ebnf(
    ArleXGrammarCompiler* compiler,
    const char* grammar,
    const char* root_rule_name,
    ArleXGrammarCompiledGrammar** out,
    char** error
);
void arle_xgrammar_compiled_grammar_free(ArleXGrammarCompiledGrammar* grammar);

int arle_xgrammar_matcher_new(
    const ArleXGrammarCompiledGrammar* grammar,
    const int32_t* override_stop_tokens,
    std::size_t override_stop_tokens_len,
    uint8_t terminate_without_stop_token,
    int32_t max_rollback_tokens,
    ArleXGrammarMatcher** out,
    char** error
);
void arle_xgrammar_matcher_free(ArleXGrammarMatcher* matcher);

int arle_xgrammar_matcher_fill_next_token_bitmask(
    ArleXGrammarMatcher* matcher,
    uint32_t* bitmask,
    std::size_t bitmask_len,
    uint8_t* need_apply,
    char** error
);
int arle_xgrammar_matcher_accept_token(
    ArleXGrammarMatcher* matcher,
    int32_t token_id,
    uint8_t* accepted,
    char** error
);
uint8_t arle_xgrammar_matcher_is_terminated(const ArleXGrammarMatcher* matcher);
uint8_t arle_xgrammar_matcher_is_completed(const ArleXGrammarMatcher* matcher);

}
