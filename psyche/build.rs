use std::path::Path;

fn main() {
    if std::env::var_os("CARGO_FEATURE_LLAMA_CUDA").is_none() {
        return;
    }

    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    for env_var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(env_var) {
            add_search_path(Path::new(&root).join("lib64"));
            add_search_path(Path::new(&root).join("lib"));
        }
    }

    for path in [
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda/lib64",
        "/usr/local/lib/ollama/cuda_v12",
        "/usr/local/lib/ollama/cuda_v13",
    ] {
        add_search_path(path);
    }

    for lib in ["cuda", "cudart", "cublas", "cublasLt"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}

fn add_search_path(path: impl AsRef<Path>) {
    let path = path.as_ref();
    if path.exists() {
        println!("cargo:rustc-link-search=native={}", path.display());
    }
}
