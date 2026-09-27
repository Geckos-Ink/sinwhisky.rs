use std::io::Write;
use std::process::{Command, Stdio};

fn cli(args: &[&str], input: &str) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_sinwhisky"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}
#[test]
fn command_line_modes_and_failures() {
    let output = cli(&["3"], "0 4");
    assert!(output.status.success());
    let samples: Vec<f64> = String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect();
    assert_eq!(samples, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    assert!(cli(&["3", "1", "1", "angular"], "0 1 0").status.success());
    assert!(cli(&["3"], "").status.success());
    for (args, input) in [
        (vec![], "0 1"),
        (vec!["-1"], "0 1"),
        (vec!["3"], "NaN"),
        (vec!["3"], "hello"),
        (vec!["3", "0"], "0 1"),
        (vec!["3", "2", "1", "angular"], "0 1 0"),
        (vec!["3", "1", "1", "unknown"], "0 1"),
    ] {
        let output = cli(&args, input);
        assert_eq!(output.status.code(), Some(2));
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
    }
}
