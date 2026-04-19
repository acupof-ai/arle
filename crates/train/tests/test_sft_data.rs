use std::{collections::HashMap, error::Error, fs, path::Path};

use tempfile::tempdir;
use tokenizers::{AddedToken, Tokenizer, models::wordlevel::WordLevel};
use train::{
    sft_data::{ChatMessage, SftExample, load_jsonl, tokenize_example},
    tokenizer::TrainTokenizer,
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
                ChatMessage {
                    role: "user".into(),
                    content: "alpha".into(),
                },
                ChatMessage {
                    role: "assistant".into(),
                    content: "beta".into(),
                },
            ],
        }
    );
    assert_eq!(
        examples[1],
        SftExample {
            messages: vec![
                ChatMessage {
                    role: "system".into(),
                    content: "policy".into(),
                },
                ChatMessage {
                    role: "assistant".into(),
                    content: "gamma".into(),
                },
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
    let tokenizer = TrainTokenizer::from_file(&tokenizer_path)?;
    let example = SftExample {
        messages: vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
            },
            ChatMessage {
                role: "assistant".into(),
                content: "hello".into(),
            },
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
