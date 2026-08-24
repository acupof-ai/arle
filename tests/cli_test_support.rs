use std::process::{Command, Output};

pub fn run_arle(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_arle"))
        .args(args)
        .output()
        .expect("spawn arle")
}

pub fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

pub fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}
