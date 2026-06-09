use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

const CRATE_NAME: &str = "llama-cpp-sys-4";
const CRATE_VERSION: &str = "0.3.1";
const PATCH_PATH: &str = "patches/llama-cpp-sys-4/0001-gemma4a-mmproj-embedding-size.patch";

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("prepare-llama-cpp-sys") => prepare_llama_cpp_sys(),
        Some(command) => Err(format!("unknown xtask command: {command}").into()),
        None => Err("usage: xtask prepare-llama-cpp-sys".into()),
    }
}

fn prepare_llama_cpp_sys() -> Result<(), Box<dyn std::error::Error>> {
    let root = workspace_root()?;
    let work_dir = root.join("target/xtask/llama-cpp-sys-fetch");
    let vendor_dir = work_dir.join("vendor");
    let patched_dir = root.join("target/patched-crates").join(CRATE_NAME);
    let fetched_dir = vendor_dir.join(format!("{CRATE_NAME}-{CRATE_VERSION}"));

    if work_dir.exists() {
        fs::remove_dir_all(&work_dir)?;
    }
    fs::create_dir_all(&work_dir)?;

    write_fetch_manifest(&work_dir)?;
    run_command(
        Command::new("cargo")
            .arg("vendor")
            .arg("--quiet")
            .arg("--versioned-dirs")
            .arg("--manifest-path")
            .arg(work_dir.join("Cargo.toml"))
            .arg(&vendor_dir)
            .stdout(Stdio::null()),
    )?;

    if !fetched_dir.exists() {
        return Err(format!("cargo vendor did not create {}", fetched_dir.display()).into());
    }

    if patched_dir.exists() {
        fs::remove_dir_all(&patched_dir)?;
    }
    copy_dir_recursive(&fetched_dir, &patched_dir)?;

    let patch = root.join(PATCH_PATH);
    run_command(
        Command::new("patch")
            .arg("--batch")
            .arg("--forward")
            .arg("-p1")
            .arg("-i")
            .arg(&patch)
            .current_dir(&patched_dir),
    )?;
    verify_llama_cpp_sys_patch(&patched_dir)?;
    verify_bindgen_prerequisites()?;

    println!(
        "prepared patched {CRATE_NAME} {CRATE_VERSION} at {}",
        patched_dir.display()
    );
    Ok(())
}

fn workspace_root() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let output = Command::new("git")
        .arg("rev-parse")
        .arg("--show-toplevel")
        .output()?;
    if !output.status.success() {
        return Err("failed to locate workspace root with git rev-parse".into());
    }
    let root = String::from_utf8(output.stdout)?;
    Ok(PathBuf::from(root.trim()))
}

fn write_fetch_manifest(dir: &Path) -> io::Result<()> {
    fs::create_dir_all(dir.join("src"))?;
    fs::write(dir.join("src/lib.rs"), "")?;
    fs::write(
        dir.join("Cargo.toml"),
        format!(
            "[package]\nname = \"llama_cpp_sys_fetch\"\nversion = \"0.0.0\"\nedition = \"2024\"\n\n[workspace]\n\n[dependencies]\n{CRATE_NAME} = \"={CRATE_VERSION}\"\n"
        ),
    )
}

fn verify_llama_cpp_sys_patch(patched_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let clip_cpp = patched_dir.join("llama.cpp/tools/mtmd/clip.cpp");
    let source = fs::read_to_string(&clip_cpp)?;
    let expected = concat!(
        "case PROJECTOR_TYPE_LFM2A:\n",
        "            return ctx->model.position_embeddings->ne[0];\n",
        "        case PROJECTOR_TYPE_GEMMA4A:\n",
        "        case PROJECTOR_TYPE_GEMMA4UA:"
    );
    if source.contains(expected) {
        Ok(())
    } else {
        Err(format!(
            "{} does not contain the Gemma 4 audio projector patch",
            clip_cpp.display()
        )
        .into())
    }
}

fn verify_bindgen_prerequisites() -> Result<(), Box<dyn std::error::Error>> {
    if clang_resource_header_exists("stdbool.h") {
        return Ok(());
    }

    Err(concat!(
        "bindgen cannot find clang's builtin C headers; llama-cpp-sys will fail ",
        "with errors like `fatal error: 'stdbool.h' file not found`.\n",
        "Install clang's development/resource headers and rerun this task.\n",
        "On Ubuntu 24.04, run: sudo apt install clang-18 libclang-18-dev\n",
        "Generic Debian/Ubuntu fallback: sudo apt install clang libclang-dev"
    )
    .into())
}

fn clang_resource_header_exists(header: &str) -> bool {
    clang_print_resource_dir()
        .into_iter()
        .chain(clang_resource_dir_candidates())
        .any(|dir| dir.join("include").join(header).exists())
}

fn clang_print_resource_dir() -> Option<PathBuf> {
    let output = Command::new("clang")
        .arg("-print-resource-dir")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8(output.stdout).ok()?;
    let path = stdout.trim();
    if path.is_empty() {
        None
    } else {
        Some(PathBuf::from(path))
    }
}

fn clang_resource_dir_candidates() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    collect_child_dirs(Path::new("/usr/lib/clang"), &mut dirs);
    collect_llvm_clang_dirs(Path::new("/usr/lib"), &mut dirs);
    dirs
}

fn collect_child_dirs(parent: &Path, dirs: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(parent) else {
        return;
    };

    for entry in entries.flatten() {
        if entry.file_type().is_ok_and(|file_type| file_type.is_dir()) {
            dirs.push(entry.path());
        }
    }
}

fn collect_llvm_clang_dirs(parent: &Path, dirs: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(parent) else {
        return;
    };

    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else {
            continue;
        };
        if !name.starts_with("llvm-")
            || !entry.file_type().is_ok_and(|file_type| file_type.is_dir())
        {
            continue;
        }
        collect_child_dirs(&entry.path().join("lib/clang"), dirs);
    }
}

fn copy_dir_recursive(from: &Path, to: &Path) -> io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let target = to.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_recursive(&entry.path(), &target)?;
        } else if file_type.is_file() {
            fs::copy(entry.path(), target)?;
        } else if file_type.is_symlink() {
            let link_target = fs::read_link(entry.path())?;
            #[cfg(unix)]
            std::os::unix::fs::symlink(link_target, target)?;
            #[cfg(windows)]
            {
                let source = entry.path();
                if source.is_dir() {
                    std::os::windows::fs::symlink_dir(link_target, target)?;
                } else {
                    std::os::windows::fs::symlink_file(link_target, target)?;
                }
            }
        }
    }
    Ok(())
}

fn run_command(command: &mut Command) -> Result<(), Box<dyn std::error::Error>> {
    let status = command.status()?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("command failed with {status:?}").into())
    }
}
