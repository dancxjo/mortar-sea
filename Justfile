set positional-arguments

default:
    @just --list

# Run the root mortar-sea CLI with forwarded args.
run *args:
    cargo run -- "$@"

# Run the inexpensive local test set: speech plus StyleTTS2 contract tests.
test:
    cargo test -p speech
    cargo test -p styletts2 --no-default-features --test contract

# Launch the Face browser server.
face:
    cargo run -- face

# Run the model management CLI with forwarded args.
models *args:
    cargo run -- models "$@"

# Open the interactive model selector.
models-menu:
    cargo run -- models

# List known model bundles.
models-list:
    cargo run -- models list

# Show selected models and local asset presence.
models-status:
    cargo run -- models status

# Print all model paths, or paths for a named model.
models-path *args:
    cargo run -- models path "$@"

# Fetch default runtime models, or a named model.
fetch *args:
    cargo run -- models fetch "$@"

models-fetch *args:
    cargo run -- models fetch "$@"

# Select the active LLM or Piper voice model.
models-use model="gemma4":
    cargo run -- models use {{ quote(model) }}

# Run the speech smoke-test CLI with forwarded args.
speak *args:
    cargo run -- speak "$@"

# Run speech synthesis through the deterministic mock backend.
speak-mock *args:
    cargo run -- speak --backend mock "$@"

# Run speech synthesis through the StyleTTS2 backend.
speak-styletts2 *args:
    cargo run -- speak --backend styletts2 "$@"

# Run speech synthesis through the Piper backend.
speak-piper *args:
    cargo run -- speak --backend piper "$@"

# Gate Voice <say> regions through Mouth with forwarded args.
mouth *args:
    cargo run -- mouth "$@"

# Open a direct terminal chat with the selected local LLM.
llm-test *args:
    cargo run -- llm-test "$@"

prepare:
    cargo run --manifest-path xtask/Cargo.toml -- prepare-llama-cpp-sys

prepare-llama-cpp-sys:
    cargo run --manifest-path xtask/Cargo.toml -- prepare-llama-cpp-sys

# Build/cache upstream Link Grammar and compare it with the local heuristic parser.
compare-lcg *args:
    cargo run --manifest-path xtask/Cargo.toml -- compare-lcg "$@"
