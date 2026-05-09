#include "xgrammar_ffi.h"

#include <dlpack/dlpack.h>
#include <xgrammar/xgrammar.h>

#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

struct ArleXGrammarCompiler {
  xgrammar::TokenizerInfo tokenizer_info;
  std::unique_ptr<xgrammar::GrammarCompiler> compiler;

  ArleXGrammarCompiler(
      xgrammar::TokenizerInfo tokenizer_info,
      int max_threads,
      bool cache_enabled,
      int64_t max_memory_bytes
  )
      : tokenizer_info(std::move(tokenizer_info)),
        compiler(std::make_unique<xgrammar::GrammarCompiler>(
            this->tokenizer_info, max_threads, cache_enabled, max_memory_bytes
        )) {}
};

struct ArleXGrammarCompiledGrammar {
  std::shared_ptr<xgrammar::CompiledGrammar> grammar;
};

struct ArleXGrammarMatcher {
  std::shared_ptr<xgrammar::CompiledGrammar> grammar;
  std::unique_ptr<xgrammar::GrammarMatcher> matcher;
};

namespace {

char* copy_message(const std::string& message) {
  auto* out = static_cast<char*>(std::malloc(message.size() + 1));
  if (out == nullptr) {
    return nullptr;
  }
  std::memcpy(out, message.c_str(), message.size() + 1);
  return out;
}

int fail(char** error, const std::string& message) {
  if (error != nullptr) {
    *error = copy_message(message);
  }
  return -1;
}

int fail(char** error, const std::exception& err) { return fail(error, err.what()); }

std::vector<std::string> copy_vocab(const char* const* encoded_vocab, std::size_t len) {
  std::vector<std::string> out;
  out.reserve(len);
  for (std::size_t i = 0; i < len; ++i) {
    out.emplace_back(encoded_vocab[i] == nullptr ? "" : encoded_vocab[i]);
  }
  return out;
}

std::optional<std::vector<int32_t>> copy_optional_i32(
    const int32_t* values, std::size_t len
) {
  if (values == nullptr || len == 0) {
    return std::nullopt;
  }
  return std::vector<int32_t>(values, values + len);
}

xgrammar::VocabType vocab_type_from_i32(int32_t value) {
  switch (value) {
    case 0:
      return xgrammar::VocabType::RAW;
    case 1:
      return xgrammar::VocabType::BYTE_FALLBACK;
    case 2:
      return xgrammar::VocabType::BYTE_LEVEL;
    default:
      throw std::invalid_argument("unknown xgrammar vocab type");
  }
}

DLTensor bitmask_tensor(uint32_t* bitmask, int64_t* shape) {
  DLTensor tensor;
  tensor.data = bitmask;
  tensor.device = DLDevice{kDLCPU, 0};
  tensor.ndim = 1;
  tensor.dtype = xgrammar::GetBitmaskDLType();
  tensor.shape = shape;
  tensor.strides = nullptr;
  tensor.byte_offset = 0;
  return tensor;
}

}  // namespace

extern "C" {

const char* arle_xgrammar_version() { return "mlc-ai/xgrammar v0.1.34"; }

void arle_xgrammar_free_error(char* message) { std::free(message); }

int32_t arle_xgrammar_bitmask_size(int32_t vocab_size) {
  return xgrammar::GetBitmaskSize(vocab_size);
}

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
) {
  try {
    if (out == nullptr) {
      return fail(error, "compiler output pointer is null");
    }
    if (encoded_vocab == nullptr && encoded_vocab_len != 0) {
      return fail(error, "encoded_vocab pointer is null");
    }
    auto vocab = copy_vocab(encoded_vocab, encoded_vocab_len);
    auto stops = copy_optional_i32(stop_token_ids, stop_token_ids_len);
    xgrammar::TokenizerInfo tokenizer_info(
        vocab,
        vocab_type_from_i32(vocab_type),
        vocab_size >= 0 ? std::optional<int>(vocab_size) : std::nullopt,
        stops,
        add_prefix_space != 0
    );
    auto handle = std::make_unique<ArleXGrammarCompiler>(
        std::move(tokenizer_info),
        max_threads,
        cache_enabled != 0,
        max_memory_bytes
    );
    *out = handle.release();
    return 0;
  } catch (const std::exception& err) {
    return fail(error, err);
  } catch (...) {
    return fail(error, "unknown xgrammar compiler construction error");
  }
}

void arle_xgrammar_compiler_free(ArleXGrammarCompiler* compiler) { delete compiler; }

int arle_xgrammar_compile_json_schema(
    ArleXGrammarCompiler* compiler,
    const char* schema,
    uint8_t strict_mode,
    ArleXGrammarCompiledGrammar** out,
    char** error
) {
  try {
    if (compiler == nullptr || out == nullptr || schema == nullptr) {
      return fail(error, "compile_json_schema received a null pointer");
    }
    auto compiled = compiler->compiler->CompileJSONSchema(
        schema,
        true,
        std::nullopt,
        std::nullopt,
        strict_mode != 0,
        std::nullopt
    );
    auto handle = std::make_unique<ArleXGrammarCompiledGrammar>();
    handle->grammar = std::make_shared<xgrammar::CompiledGrammar>(std::move(compiled));
    *out = handle.release();
    return 0;
  } catch (const std::exception& err) {
    return fail(error, err);
  } catch (...) {
    return fail(error, "unknown xgrammar JSON schema compile error");
  }
}

int arle_xgrammar_compile_ebnf(
    ArleXGrammarCompiler* compiler,
    const char* grammar,
    const char* root_rule_name,
    ArleXGrammarCompiledGrammar** out,
    char** error
) {
  try {
    if (compiler == nullptr || out == nullptr || grammar == nullptr) {
      return fail(error, "compile_ebnf received a null pointer");
    }
    auto compiled = compiler->compiler->CompileGrammar(
        grammar,
        root_rule_name == nullptr ? "root" : root_rule_name
    );
    auto handle = std::make_unique<ArleXGrammarCompiledGrammar>();
    handle->grammar = std::make_shared<xgrammar::CompiledGrammar>(std::move(compiled));
    *out = handle.release();
    return 0;
  } catch (const std::exception& err) {
    return fail(error, err);
  } catch (...) {
    return fail(error, "unknown xgrammar EBNF compile error");
  }
}

void arle_xgrammar_compiled_grammar_free(ArleXGrammarCompiledGrammar* grammar) {
  delete grammar;
}

int arle_xgrammar_matcher_new(
    const ArleXGrammarCompiledGrammar* grammar,
    const int32_t* override_stop_tokens,
    std::size_t override_stop_tokens_len,
    uint8_t terminate_without_stop_token,
    int32_t max_rollback_tokens,
    ArleXGrammarMatcher** out,
    char** error
) {
  try {
    if (grammar == nullptr || out == nullptr) {
      return fail(error, "matcher_new received a null pointer");
    }
    auto stops = copy_optional_i32(override_stop_tokens, override_stop_tokens_len);
    auto handle = std::make_unique<ArleXGrammarMatcher>();
    handle->grammar = grammar->grammar;
    handle->matcher = std::make_unique<xgrammar::GrammarMatcher>(
        *handle->grammar,
        stops,
        terminate_without_stop_token != 0,
        max_rollback_tokens
    );
    *out = handle.release();
    return 0;
  } catch (const std::exception& err) {
    return fail(error, err);
  } catch (...) {
    return fail(error, "unknown xgrammar matcher construction error");
  }
}

void arle_xgrammar_matcher_free(ArleXGrammarMatcher* matcher) { delete matcher; }

int arle_xgrammar_matcher_fill_next_token_bitmask(
    ArleXGrammarMatcher* matcher,
    uint32_t* bitmask,
    std::size_t bitmask_len,
    uint8_t* need_apply,
    char** error
) {
  try {
    if (matcher == nullptr || bitmask == nullptr || need_apply == nullptr) {
      return fail(error, "fill_next_token_bitmask received a null pointer");
    }
    int64_t shape[1] = {static_cast<int64_t>(bitmask_len)};
    auto tensor = bitmask_tensor(bitmask, shape);
    *need_apply = matcher->matcher->FillNextTokenBitmask(&tensor) ? 1 : 0;
    return 0;
  } catch (const std::exception& err) {
    return fail(error, err);
  } catch (...) {
    return fail(error, "unknown xgrammar bitmask fill error");
  }
}

int arle_xgrammar_matcher_accept_token(
    ArleXGrammarMatcher* matcher,
    int32_t token_id,
    uint8_t* accepted,
    char** error
) {
  try {
    if (matcher == nullptr || accepted == nullptr) {
      return fail(error, "accept_token received a null pointer");
    }
    *accepted = matcher->matcher->AcceptToken(token_id) ? 1 : 0;
    return 0;
  } catch (const std::exception& err) {
    return fail(error, err);
  } catch (...) {
    return fail(error, "unknown xgrammar accept_token error");
  }
}

uint8_t arle_xgrammar_matcher_is_terminated(const ArleXGrammarMatcher* matcher) {
  return matcher != nullptr && matcher->matcher->IsTerminated() ? 1 : 0;
}

uint8_t arle_xgrammar_matcher_is_completed(const ArleXGrammarMatcher* matcher) {
  return matcher != nullptr && matcher->matcher->IsCompleted() ? 1 : 0;
}

}  // extern "C"
