default:
    @just --list

prepare-llama-cpp-sys:
    cargo run --manifest-path xtask/Cargo.toml -- prepare-llama-cpp-sys
