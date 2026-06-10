use std::{
    env, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};
use syntax::parse_english_link_grammar;

mod segment {
    use super::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum TerminalPunctuation {
        Period,
        Question,
        Exclamation,
    }
}

#[allow(dead_code)]
#[path = "../../speech/src/syntax.rs"]
mod syntax;

use segment::TerminalPunctuation;

const CRATE_NAME: &str = "llama-cpp-sys-4";
const CRATE_VERSION: &str = "0.3.1";
const PATCH_PATH: &str = "patches/llama-cpp-sys-4/0001-gemma4a-mmproj-embedding-size.patch";
const LINK_GRAMMAR_VERSION: &str = "5.13.0";
const LINK_GRAMMAR_TARBALL_URL: &str =
    "https://www.gnucash.org/link-grammar/downloads/5.13.0/link-grammar-5.13.0.tar.gz";

const DEFAULT_LCG_SAMPLES: &[&str] = &[
    // English sentences embedded in the upstream C/C++ tests:
    // tests/multi-dict.cc, tests/multi-thread.cc, tests/dict-reopen.cc,
    // and tests/mem-leak.cc. Several intentionally produce many linkages.
    "Frank felt vindicated when his long time friend Bill revealed that he was the winner of the competition.",
    "Logorrhea, or excessive and often incoherent talkativeness or wordiness, is a social disease.",
    "It was covered with bites.",
    "I have no idea what that is.",
    "His shout had been involuntary, something anybody might have done.",
    "Trump, Ryan and McConnell are using the budget process to pay for the GOP’s $1.5 trillion tax scam.",
    "We ate popcorn and watched movies on TV for three days.",
    "Sweat stood on his brow, fury was bright in his one good eye.",
    "One of the things you do when you stop your bicycle is apply the brake.",
    "The line extends 10 miles offshore.",
    "He is the kind of person who would do that.",
    "The mystery of the Nixon tapes was never solved.",
    "Perhaps it is and perhaps it isnt.",
];

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
        Some("compare-lcg") => compare_lcg(args.collect()),
        Some(command) => Err(format!("unknown xtask command: {command}").into()),
        None => Err("usage: xtask <prepare-llama-cpp-sys|compare-lcg>".into()),
    }
}

fn compare_lcg(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let options = parse_compare_lcg_options(args)?;
    let root = workspace_root()?;
    let reference = ensure_reference_link_parser(&root, options.rebuild, options.refresh)?;

    println!("reference source: {}", reference.source_dir.display());
    println!("reference binary: {}", reference.parser.display());
    println!("reference version: {LINK_GRAMMAR_VERSION}");
    println!();

    for sentence in options.samples {
        let official = run_official_link_parser(&reference, &sentence)?;
        let ours = run_our_link_parser(&sentence);
        println!(">>> {sentence}");
        println!(
            "official implementation: {}{} in {}",
            pass_label(official.accepted),
            official
                .linkage_count
                .map(|count| format!(" ({count} linkages)"))
                .unwrap_or_default(),
            format_duration(official.elapsed)
        );
        println!(
            "our implementation: {} ({} typed links) in {}",
            pass_label(ours.accepted),
            ours.link_count,
            format_duration(ours.elapsed)
        );
        if official.accepted != ours.accepted {
            println!("result: mismatch");
        }
        println!();
    }

    Ok(())
}

#[derive(Debug)]
struct CompareLcgOptions {
    rebuild: bool,
    refresh: bool,
    samples: Vec<String>,
}

fn parse_compare_lcg_options(
    args: Vec<String>,
) -> Result<CompareLcgOptions, Box<dyn std::error::Error>> {
    let mut rebuild = false;
    let mut refresh = false;
    let mut sentence_parts = Vec::new();
    for arg in args {
        match arg.as_str() {
            "--rebuild" => rebuild = true,
            "--refresh" => {
                refresh = true;
                rebuild = true;
            }
            "--help" | "-h" => {
                return Err(concat!(
                    "usage: xtask compare-lcg [--refresh] [--rebuild] [sentence...]\n",
                    "no sentence runs the built-in benchmark samples"
                )
                .into());
            }
            "--" => {}
            _ => sentence_parts.push(arg),
        }
    }

    let samples = if sentence_parts.is_empty() {
        DEFAULT_LCG_SAMPLES
            .iter()
            .map(|sample| (*sample).to_string())
            .collect()
    } else {
        vec![sentence_parts.join(" ")]
    };

    Ok(CompareLcgOptions {
        rebuild,
        refresh,
        samples,
    })
}

#[derive(Debug)]
struct LinkGrammarReference {
    source_dir: PathBuf,
    parser: PathBuf,
    dictionary_dir: PathBuf,
}

fn ensure_reference_link_parser(
    root: &Path,
    rebuild: bool,
    refresh: bool,
) -> Result<LinkGrammarReference, Box<dyn std::error::Error>> {
    let work_dir = root.join("target/xtask/link-grammar");
    let source_dir = work_dir.join(format!("link-grammar-{LINK_GRAMMAR_VERSION}"));
    let parser = source_dir.join("link-parser/link-parser");
    let dictionary_dir = source_dir.join("data/en");
    let archive = work_dir.join(format!("link-grammar-{LINK_GRAMMAR_VERSION}.tar.gz"));

    if refresh && source_dir.exists() {
        fs::remove_dir_all(&source_dir)?;
    }
    if refresh && archive.exists() {
        fs::remove_file(&archive)?;
    }
    if !source_dir.exists() {
        fetch_link_grammar_release(&work_dir, &archive)?;
        unpack_link_grammar_release(&work_dir, &archive)?;
    }

    if rebuild || !parser.exists() {
        build_link_grammar_source(&source_dir)?;
    }
    if !parser.exists() {
        return Err(format!("reference build did not create {}", parser.display()).into());
    }
    if !dictionary_dir.exists() {
        return Err(format!(
            "reference dictionary missing at {}",
            dictionary_dir.display()
        )
        .into());
    }

    Ok(LinkGrammarReference {
        source_dir,
        parser,
        dictionary_dir,
    })
}

fn fetch_link_grammar_release(
    work_dir: &Path,
    archive: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    verify_program("curl", "download the Link Grammar release tarball")?;
    verify_program("tar", "unpack the Link Grammar release tarball")?;
    fs::create_dir_all(work_dir)?;
    if archive.exists() {
        return Ok(());
    }
    run_command(
        Command::new("curl")
            .arg("--fail")
            .arg("--location")
            .arg("--output")
            .arg(archive)
            .arg(LINK_GRAMMAR_TARBALL_URL),
    )
}

fn unpack_link_grammar_release(
    work_dir: &Path,
    archive: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    run_command(
        Command::new("tar")
            .arg("-xzf")
            .arg(archive)
            .arg("-C")
            .arg(work_dir),
    )
}

fn build_link_grammar_source(source_dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    verify_link_grammar_build_prerequisites()?;
    let fake_lex = write_fake_lex_shim(source_dir)?;
    if !source_dir.join("configure").exists() {
        return Err(format!(
            "{} does not contain configure; expected the official release tarball layout",
            source_dir.display()
        )
        .into());
    }

    run_command(
        Command::new("./configure")
            .arg("--disable-java-bindings")
            .arg("--disable-python-bindings")
            .env("LEX", &fake_lex)
            .current_dir(source_dir),
    )?;

    let jobs = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(2)
        .to_string();
    run_command(
        Command::new("make")
            .arg("-C")
            .arg("link-grammar")
            .arg("-j")
            .arg(&jobs)
            .arg("liblink-grammar.la")
            .current_dir(source_dir),
    )?;
    run_command(
        Command::new("make")
            .arg("-C")
            .arg("link-parser")
            .arg("-j")
            .arg(jobs)
            .arg("link-parser")
            .current_dir(source_dir),
    )
}

fn verify_link_grammar_build_prerequisites() -> Result<(), Box<dyn std::error::Error>> {
    verify_program("make", "build the Link Grammar reference parser")?;
    Ok(())
}

fn write_fake_lex_shim(source_dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let shim = source_dir.join("xtask-fake-lex");
    fs::write(
        &shim,
        concat!(
            "#!/bin/sh\n",
            "out=lex.yy.c\n",
            "prev=\n",
            "for arg in \"$@\"; do\n",
            "  if [ \"$prev\" = \"-o\" ]; then out=\"$arg\"; prev=; continue; fi\n",
            "  case \"$arg\" in\n",
            "    -o) prev=-o ;;\n",
            "    -o*) out=\"${arg#-o}\" ;;\n",
            "  esac\n",
            "done\n",
            "cat > \"$out\" <<'EOF'\n",
            "char *yytext;\n",
            "int yylex(void) { return 0; }\n",
            "int yywrap(void) { return 1; }\n",
            "int main(void) { return yylex(); }\n",
            "EOF\n",
        ),
    )?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(&shim)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&shim, permissions)?;
    }
    Ok(shim)
}

fn verify_program(program: &str, purpose: &str) -> Result<(), Box<dyn std::error::Error>> {
    if program_exists(program) {
        Ok(())
    } else {
        Err(format!("`{program}` is required to {purpose}").into())
    }
}

fn program_exists(program: &str) -> bool {
    let Some(path) = env::var_os("PATH") else {
        return false;
    };
    env::split_paths(&path).any(|dir| dir.join(program).is_file())
}

#[derive(Debug)]
struct OfficialRun {
    accepted: bool,
    linkage_count: Option<usize>,
    elapsed: Duration,
}

fn run_official_link_parser(
    reference: &LinkGrammarReference,
    sentence: &str,
) -> Result<OfficialRun, Box<dyn std::error::Error>> {
    let start = Instant::now();
    let mut child = Command::new(&reference.parser)
        .arg(&reference.dictionary_dir)
        .arg("-verbosity=1")
        .arg("-graphics=0")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    {
        let stdin = child
            .stdin
            .as_mut()
            .ok_or("failed to open link-parser stdin")?;
        writeln!(stdin, "{sentence}")?;
        writeln!(stdin, "!quit")?;
    }
    let output = child.wait_with_output()?;
    let elapsed = start.elapsed();
    if !output.status.success() {
        return Err(format!(
            "link-parser failed for `{sentence}`\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
        .into());
    }

    let combined_output = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let linkage_count = parse_official_linkage_count(&combined_output);
    Ok(OfficialRun {
        accepted: linkage_count.is_some_and(|count| count > 0),
        linkage_count,
        elapsed,
    })
}

fn parse_official_linkage_count(output: &str) -> Option<usize> {
    for line in output.lines() {
        let Some(rest) = line.trim_start().strip_prefix("Found ") else {
            continue;
        };
        let digits = rest
            .chars()
            .take_while(|character| character.is_ascii_digit())
            .collect::<String>();
        if digits.is_empty() {
            continue;
        }
        if let Ok(count) = digits.parse() {
            return Some(count);
        }
    }
    None
}

#[derive(Debug)]
struct OurRun {
    accepted: bool,
    link_count: usize,
    elapsed: Duration,
}

fn run_our_link_parser(sentence: &str) -> OurRun {
    let words = tokenize_lcg_sentence(sentence);
    let terminal = terminal_for_sentence(sentence);
    let start = Instant::now();
    let analysis = parse_english_link_grammar(&words, terminal);
    let elapsed = start.elapsed();
    let link_count = analysis
        .primary_parse()
        .map(|parse| parse.links.len())
        .unwrap_or_default();
    OurRun {
        accepted: link_count > 0 || words.len() <= 1,
        link_count,
        elapsed,
    }
}

fn tokenize_lcg_sentence(sentence: &str) -> Vec<String> {
    sentence
        .split_whitespace()
        .map(|word| {
            word.trim_matches(|character: char| !character.is_alphabetic() && character != '\'')
                .to_string()
        })
        .filter(|word| !word.is_empty())
        .collect()
}

fn terminal_for_sentence(sentence: &str) -> Option<TerminalPunctuation> {
    match sentence.trim().chars().next_back()? {
        '?' => Some(TerminalPunctuation::Question),
        '!' => Some(TerminalPunctuation::Exclamation),
        '.' | '…' => Some(TerminalPunctuation::Period),
        _ => None,
    }
}

fn pass_label(accepted: bool) -> &'static str {
    if accepted { "accepted" } else { "rejected" }
}

fn format_duration(duration: Duration) -> String {
    let seconds = duration.as_secs_f64();
    if seconds < 0.001 {
        format!("{seconds:.6}s")
    } else {
        format!("{seconds:.3}s")
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
