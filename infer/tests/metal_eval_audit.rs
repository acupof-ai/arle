use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("infer crate should live under repo root")
        .to_path_buf()
}

fn strip_line_comment(line: &str) -> &str {
    line.split_once("//").map_or(line, |(code, _)| code)
}

fn materialize_count_for_line(line: &str) -> usize {
    let async_eval_count = line.match_indices("async_eval(").count();
    let eval_count = line
        .match_indices("eval(")
        .filter(|(idx, _)| !line[..*idx].ends_with("async_"))
        .count();
    let item_count = line.match_indices(".item(").count();
    async_eval_count + eval_count + item_count
}

fn materialize_count(path: &Path) -> usize {
    let contents = fs::read_to_string(path).expect("read audit target");
    contents
        .lines()
        .map(strip_line_comment)
        .map(materialize_count_for_line)
        .sum()
}

fn collect_source_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read source dir") {
        let path = entry.expect("read dir entry").path();
        if path.is_dir() {
            collect_source_files(&path, out);
            continue;
        }

        if matches!(
            path.extension().and_then(|ext| ext.to_str()),
            Some("rs" | "cpp" | "h" | "hpp")
        ) {
            out.push(path);
        }
    }
}

fn audit_paths() -> Vec<PathBuf> {
    let root = repo_root();
    let mut paths = Vec::new();
    collect_source_files(&root.join("infer/src/backend/metal"), &mut paths);
    collect_source_files(&root.join("crates/mlx-sys/src"), &mut paths);
    paths.sort();
    paths
}

fn relative(path: &Path) -> String {
    path.strip_prefix(repo_root())
        .expect("audit path should be under repo root")
        .to_string_lossy()
        .replace('\\', "/")
}

#[test]
fn metal_scheduler_runtime_has_no_direct_materialize_boundary() {
    let root = repo_root();
    for rel in [
        "infer/src/backend/metal/runtime.rs",
        "infer/src/backend/metal/scheduler.rs",
        "infer/src/backend/metal.rs",
    ] {
        let count = materialize_count(&root.join(rel));
        assert_eq!(
            count, 0,
            "{rel} should not call eval/async_eval/.item directly"
        );
    }
}

#[test]
fn metal_materialize_boundaries_stay_classified() {
    let expected = BTreeMap::from([
        ("crates/mlx-sys/src/lib.rs", 2usize),
        ("crates/mlx-sys/src/mlx_bridge.cpp", 4),
        ("crates/mlx-sys/src/mlx_qwen35_model.cpp", 10),
        ("infer/src/backend/metal/dflash.rs", 6),
        ("infer/src/backend/metal/gdr.rs", 10),
        ("infer/src/backend/metal/generate.rs", 2),
        ("infer/src/backend/metal/loader.rs", 2),
        ("infer/src/backend/metal/mlx.rs", 16),
        ("infer/src/backend/metal/ops.rs", 2),
        ("infer/src/backend/metal/qwen35.rs", 10),
        ("infer/src/backend/metal/request_state.rs", 27),
        ("infer/src/backend/metal/weights.rs", 1),
    ]);

    let mut actual = BTreeMap::new();
    for path in audit_paths() {
        if path.file_name().and_then(|name| name.to_str()) == Some("tests.rs") {
            continue;
        }
        let count = materialize_count(&path);
        if count > 0 {
            actual.insert(relative(&path), count);
        }
    }

    let expected_paths = expected.keys().copied().collect::<BTreeSet<_>>();
    let actual_paths = actual.keys().map(String::as_str).collect::<BTreeSet<_>>();
    assert_eq!(
        actual_paths, expected_paths,
        "new Metal materialize boundary file needs docs/experience classification"
    );

    for (path, expected_count) in expected {
        assert_eq!(
            actual.get(path).copied(),
            Some(expected_count),
            "{path} materialize count changed; update the audit classification"
        );
    }
}
