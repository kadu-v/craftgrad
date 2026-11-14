use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::Result;
use crate::uop::{Arg, Ops, UOp};

static NEXT: AtomicUsize = AtomicUsize::new(0);

pub fn expression(root: &UOp) -> String {
    let root = root.rewrite(|_| None);
    let mut terms = vec![];

    for node in root.postorder() {
        let term = match node.op {
            Ops::Const => match node.arg {
                Arg::Float(value) => format!("chibi_float({}u)", (value.value() as f32).to_bits()),
            },
        };
        terms.push((node, term));
    }
    terms
        .pop()
        .expect("[Internal compiler error] Terms must have one element at least")
        .1
}

pub fn source(root: &UOp) -> String {
    format!(
        "#include <stdint.h>\n#include <string.h>\nstatic float chibi_float(uint32_t bits) {{ float value; memcpy(&value, &bits, 4); return value; }}\n#include <stdio.h>\nint main(void) {{ printf(\"%.9g\\n\", (double)({})); }}\n",
        expression(root)
    )
}

pub fn run(root: &UOp) -> Result<f32> {
    let source = source(root);

    let target_dir = std::env::var_os("CARGO_TARGET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target"));

    let dir = target_dir.join(format!(
        "chibigrad-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir(&dir).map_err(|e| e.to_string())?;
    let result = (|| {
        let input = dir.join("scalar.c");
        let binary = dir.join("scalar");
        fs::write(&input, source).map_err(|e| e.to_string())?;
        let compiler = std::env::var_os("CC").unwrap_or_else(|| {
            if cfg!(target_os = "macos") {
                "/usr/bin/clang".into()
            } else {
                "clang".into()
            }
        });
        let compile = Command::new(compiler)
            .args(["-std=c11", "-O2"])
            .arg(&input)
            .arg("-o")
            .arg(&binary)
            .output()
            .map_err(|e| e.to_string())?;
        if !compile.status.success() {
            return Err(String::from_utf8_lossy(&compile.stderr).into());
        }
        let output = Command::new(binary).output().map_err(|e| e.to_string())?;
        if !output.status.success() {
            return Err("scalar executable failed".into());
        }
        String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<f32>()
            .map_err(|e| e.to_string())
    })();
    fs::remove_dir_all(&dir).map_err(|e| e.to_string())?;
    result
}
