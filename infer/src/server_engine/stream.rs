#[cfg(any(feature = "metal", feature = "cpu", test))]
use std::path::Path;

#[cfg(any(feature = "metal", feature = "cpu", test))]
use super::FinishReason;

/// Truncate at the first occurrence of any stop string (OpenAI-compatible).
/// Returns the prefix of `text` up to (but not including) the earliest stop.
#[cfg(any(feature = "metal", feature = "cpu", test))]
pub(super) fn truncate_at_first_stop(text: &str, stops: &[String]) -> Option<String> {
    let mut earliest = None::<usize>;
    for s in stops {
        let s = s.as_str();
        if s.is_empty() {
            continue;
        }
        if let Some(pos) = text.find(s) {
            earliest = Some(match earliest {
                None => pos,
                Some(e) => std::cmp::min(e, pos),
            });
        }
    }
    earliest.map(|pos| text[..pos].to_string())
}

#[cfg(any(feature = "metal", feature = "cpu", test))]
pub(super) fn model_id_from_path(model_path: &str) -> String {
    Path::new(model_path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or(model_path)
        .to_string()
}

#[cfg(any(feature = "metal", feature = "cpu"))]
pub(super) fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(msg) => *msg,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(msg) => (*msg).to_string(),
            Err(_) => "unknown panic payload".to_string(),
        },
    }
}

#[cfg(any(feature = "metal", feature = "cpu", test))]
pub(super) fn parse_finish_reason(finish_reason: &str) -> FinishReason {
    match finish_reason {
        "length" => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}
