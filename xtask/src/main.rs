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
        Command::new("git")
            .arg("apply")
            .arg(&patch)
            .current_dir(&patched_dir),
    )?;

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
