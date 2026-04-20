use std::{collections::HashMap, error::Error, fs, path::Path};

use tempfile::tempdir;
use tokenizers::{AddedToken, Tokenizer, models::wordlevel::WordLevel};
use train::{
    sft_data::{ChatMessage, SftExample, ToolCall, load_jsonl, tokenize_example},
    tokenizer::ChatTokenizer,
};

type TestResult<T = ()> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

#[test]
fn parse_jsonl_basic() -> TestResult {
    let dir = tempdir()?;
    let path = dir.path().join("sample.jsonl");
    fs::write(
        &path,
        concat!(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"alpha\"},{\"role\":\"assistant\",\"content\":\"beta\"}]}\n",
            "\n",
            "{\"messages\":[{\"role\":\"system\",\"content\":\"policy\"},{\"role\":\"assistant\",\"content\":\"gamma\"}]}\n",
        ),
    )?;

    let examples = load_jsonl(&path)?;
    assert_eq!(examples.len(), 2);
    assert_eq!(
        examples[0],
        SftExample {
            messages: vec![
                ChatMessage::user("alpha"),
                ChatMessage::assistant("beta", vec![]),
            ],
        }
    );
    assert_eq!(
        examples[1],
        SftExample {
            messages: vec![
                ChatMessage::system("policy"),
                ChatMessage::assistant("gamma", vec![]),
            ],
        }
    );

    Ok(())
}

#[test]
fn tokenize_masks_non_assistant() -> TestResult {
    let dir = tempdir()?;
    let tokenizer_path = dir.path().join("tokenizer.json");
    write_test_tokenizer(&tokenizer_path)?;
    let tokenizer = ChatTokenizer::from_file(&tokenizer_path)?;
    let example = SftExample {
        messages: vec![
            ChatMessage::user("hi"),
            ChatMessage::assistant("hello", vec![]),
        ],
    };

    let tokenized = tokenize_example(&example, &tokenizer, 64)?;
    assert_eq!(tokenized.input_ids.len(), tokenized.labels.len());

    let user_prefix_len = tokenizer.encode("<|im_start|>user\n", false)?.len();
    let user_content_len = tokenizer.encode("hi", false)?.len();
    let end_len = tokenizer.encode("<|im_end|>", false)?.len();
    let newline_len = tokenizer.encode("\n", false)?.len();
    let assistant_prefix_len = tokenizer.encode("<|im_start|>assistant\n", false)?.len();
    let assistant_content_len = tokenizer.encode("hello", false)?.len();

    let assistant_body_start =
        user_prefix_len + user_content_len + end_len + newline_len + assistant_prefix_len;
    let assistant_body_end = assistant_body_start + assistant_content_len + end_len;

    assert!(
        tokenized.labels[..assistant_body_start]
            .iter()
            .all(|&label| label == -100)
    );
    assert!(
        tokenized.labels[assistant_body_start..assistant_body_end]
            .iter()
            .all(|&label| label != -100)
    );
    assert!(
        tokenized.labels[assistant_body_end..]
            .iter()
            .all(|&label| label == -100)
    );

    Ok(())
}

#[test]
fn parse_jsonl_tool_calls_and_tool_results() -> TestResult {
    let dir = tempdir()?;
    let path = dir.path().join("sample.jsonl");
    fs::write(
        &path,
        concat!(
            "{\"messages\":[",
            "{\"role\":\"user\",\"content\":\"ask\"},",
            "{\"role\":\"assistant\",\"content\":null,\"tool_calls\":[{\"name\":\"shell\",\"arguments\":{\"command\":\"pwd\"}}]},",
            "{\"role\":\"tool\",\"content\":\"cwd\",\"tool_call_id\":\"call_1\",\"name\":\"shell\"},",
            "{\"role\":\"assistant\",\"content\":\"done\"}",
            "]}\n",
        ),
    )?;

    let examples = load_jsonl(&path)?;
    assert_eq!(examples.len(), 1);
    assert_eq!(
        examples[0],
        SftExample {
            messages: vec![
                ChatMessage::user("ask"),
                ChatMessage::assistant(
                    "",
                    vec![ToolCall::new(
                        "shell",
                        serde_json::json!({ "command": "pwd" })
                    )],
                ),
                ChatMessage::tool_result("shell", "cwd"),
                ChatMessage::assistant("done", vec![]),
            ],
        }
    );

    Ok(())
}

#[test]
fn tokenize_masks_tool_turns_and_labels_assistant_tool_calls() -> TestResult {
    let dir = tempdir()?;
    let tokenizer_path = dir.path().join("tokenizer.json");
    write_tool_turn_tokenizer(&tokenizer_path)?;
    let tokenizer = ChatTokenizer::from_file(&tokenizer_path)?;

    let assistant_payload = serde_json::to_string(&serde_json::json!({
        "name": "shell",
        "arguments": { "command": "pwd" }
    }))?;
    let user_turn = "<|im_start|>user\nask<|im_end|>\n";
    let assistant_tool_turn = format!(
        "<|im_start|>assistant\n\n<tool_call>\n{assistant_payload}\n</tool_call><|im_end|>\n"
    );
    let tool_turn = "<|im_start|>tool\n<tool_response>\ncwd\n</tool_response><|im_end|>\n";
    let final_turn = "<|im_start|>assistant\ndone<|im_end|>\n";

    let example = SftExample {
        messages: vec![
            ChatMessage::user("ask"),
            ChatMessage::assistant(
                "",
                vec![ToolCall::new(
                    "shell",
                    serde_json::json!({ "command": "pwd" }),
                )],
            ),
            ChatMessage::tool_result("shell", "cwd"),
            ChatMessage::assistant("done", vec![]),
        ],
    };

    let tokenized = tokenize_example(&example, &tokenizer, 128)?;
    let user_len = tokenizer.encode(user_turn, false)?.len();
    let assistant_len = tokenizer.encode(&assistant_tool_turn, false)?.len();
    let tool_len = tokenizer.encode(tool_turn, false)?.len();
    let final_len = tokenizer.encode(final_turn, false)?.len();

    let user_range = 0..user_len;
    let assistant_range = user_len..user_len + assistant_len;
    let tool_range = user_len + assistant_len..user_len + assistant_len + tool_len;
    let final_range =
        user_len + assistant_len + tool_len..user_len + assistant_len + tool_len + final_len;

    assert!(
        tokenized.labels[user_range]
            .iter()
            .all(|&label| label == -100)
    );
    assert!(
        tokenized.labels[assistant_range]
            .iter()
            .any(|&label| label != -100)
    );
    assert!(
        tokenized.labels[tool_range]
            .iter()
            .all(|&label| label == -100)
    );
    assert!(
        tokenized.labels[final_range]
            .iter()
            .any(|&label| label != -100)
    );

    Ok(())
}

fn write_test_tokenizer(path: &Path) -> TestResult {
    let vocab = HashMap::from([
        ("[UNK]".to_string(), 0),
        ("<|im_start|>user\n".to_string(), 1),
        ("<|im_start|>assistant\n".to_string(), 2),
        ("<|im_start|>system\n".to_string(), 3),
        ("<|im_end|>".to_string(), 4),
        ("\n".to_string(), 5),
        ("hi".to_string(), 6),
        ("hello".to_string(), 7),
        ("alpha".to_string(), 8),
        ("beta".to_string(), 9),
        ("policy".to_string(), 10),
        ("gamma".to_string(), 11),
    ]);
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.add_special_tokens(&[
        AddedToken::from("<|im_start|>user\n", true),
        AddedToken::from("<|im_start|>assistant\n", true),
        AddedToken::from("<|im_start|>system\n", true),
        AddedToken::from("<|im_end|>", true),
    ]);
    tokenizer.save(path, false)?;
    Ok(())
}

fn write_tool_turn_tokenizer(path: &Path) -> TestResult {
    let assistant_payload = serde_json::to_string(&serde_json::json!({
        "name": "shell",
        "arguments": { "command": "pwd" }
    }))?;
    let vocab = HashMap::from([
        ("[UNK]".to_string(), 0),
        ("<|im_start|>user\nask<|im_end|>\n".to_string(), 1),
        (
            format!(
                "<|im_start|>assistant\n\n<tool_call>\n{assistant_payload}\n</tool_call><|im_end|>\n"
            ),
            2,
        ),
        (
            "<|im_start|>tool\n<tool_response>\ncwd\n</tool_response><|im_end|>\n".to_string(),
            3,
        ),
        ("<|im_start|>assistant\ndone<|im_end|>\n".to_string(), 4),
    ]);
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.add_special_tokens(&[
        AddedToken::from("<|im_start|>user\nask<|im_end|>\n", true),
        AddedToken::from(
            format!("<|im_start|>assistant\n\n<tool_call>\n{assistant_payload}\n</tool_call><|im_end|>\n"),
            true,
        ),
        AddedToken::from(
            "<|im_start|>tool\n<tool_response>\ncwd\n</tool_response><|im_end|>\n",
            true,
        ),
        AddedToken::from("<|im_start|>assistant\ndone<|im_end|>\n", true),
    ]);
    tokenizer.save(path, false)?;
    Ok(())
}
