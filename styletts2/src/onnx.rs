use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::{DynTensorValueType, Tensor};

use crate::backend::{StyleTts2Backend, StyleTts2Error, StyleTts2SynthesisOutput};
use crate::request::StyleTts2SynthesisRequest;
use crate::symbols::{StyleTts2SymbolMapper, StyleTts2SymbolSequence, styletts2_en_us_symbol_set};

const SAMPLE_RATE_HZ: u32 = 24_000;
const TEXT_ENCODER_ONNX: &str =
    "91473db52725b0c3b8387537979a2f42f0da82836e50902503a877c610864ad6.onnx";
const DECODER_ONNX: &str = "99e40b35027e96a247c8e1f359d2f99d3cd6e93afec2e0f4a15f72dd7b79d457.onnx";
const STYLE_VECTOR_DIMS: usize = 256;
const HIDDEN_DIMS: i64 = 768;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StyleTts2OnnxPaths {
    pub text_encoder: PathBuf,
    pub decoder: PathBuf,
}

impl StyleTts2OnnxPaths {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Self {
        let model_dir = model_dir.as_ref();
        Self {
            text_encoder: model_dir.join(TEXT_ENCODER_ONNX),
            decoder: model_dir.join(DECODER_ONNX),
        }
    }
}

pub struct StyleTts2OnnxBackend {
    text_encoder: Session,
    decoder: Session,
    style_vector: Vec<f32>,
    speed: f64,
}

impl StyleTts2OnnxBackend {
    pub fn load(paths: StyleTts2OnnxPaths) -> Result<Self, StyleTts2Error> {
        ensure_file(&paths.text_encoder, "StyleTTS2 text encoder")?;
        ensure_file(&paths.decoder, "StyleTTS2 decoder")?;
        initialize_ort_runtime()?;

        Ok(Self {
            text_encoder: load_session(&paths.text_encoder, "StyleTTS2 text encoder")?,
            decoder: load_session(&paths.decoder, "StyleTTS2 decoder")?,
            style_vector: vec![0.0; STYLE_VECTOR_DIMS],
            speed: 1.0,
        })
    }

    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self, StyleTts2Error> {
        Self::load(StyleTts2OnnxPaths::from_model_dir(model_dir))
    }

    pub fn with_style_vector(mut self, style_vector: Vec<f32>) -> Result<Self, StyleTts2Error> {
        if style_vector.len() != STYLE_VECTOR_DIMS {
            return Err(invalid_output(format!(
                "StyleTTS2 style vector must have {STYLE_VECTOR_DIMS} values, got {}",
                style_vector.len()
            )));
        }
        if !style_vector.iter().all(|value| value.is_finite()) {
            return Err(invalid_output(
                "StyleTTS2 style vector contains non-finite values",
            ));
        }
        self.style_vector = style_vector;
        Ok(self)
    }

    pub fn with_speed(mut self, speed: f64) -> Result<Self, StyleTts2Error> {
        if !speed.is_finite() || speed <= 0.0 {
            return Err(invalid_output(format!(
                "StyleTTS2 speed must be finite and positive, got {speed}"
            )));
        }
        self.speed = speed;
        Ok(self)
    }
}

impl StyleTts2Backend for StyleTts2OnnxBackend {
    fn synthesize(
        &mut self,
        request: &StyleTts2SynthesisRequest,
    ) -> Result<StyleTts2SynthesisOutput, StyleTts2Error> {
        let lowered = styletts2_en_us_symbol_set().lower(&request.utterance_plan)?;
        let token_ids = styletts2_token_ids(&lowered)?;
        if token_ids.is_empty() {
            return Ok(StyleTts2SynthesisOutput {
                sample_rate_hz: SAMPLE_RATE_HZ,
                pcm_mono_f32: Vec::new(),
                realized_utterance: None,
            });
        }

        let token_len = i64::try_from(token_ids.len())
            .map_err(|_| invalid_output("StyleTTS2 token sequence is too long"))?;
        let encoder_input = Tensor::from_array((vec![1_i64, token_len], token_ids.clone()))
            .map_err(|error| backend_error(format!("failed to build text encoder input: {error}")))?
            .upcast();
        let encoder_outputs = self
            .text_encoder
            .run(vec![("a".to_string(), encoder_input)])
            .map_err(|error| backend_error(format!("StyleTTS2 text encoder failed: {error}")))?;
        let (encoder_shape, encoder_values) = extract_f32_tensor(&encoder_outputs, "z")?;
        if encoder_shape.len() != 3 || encoder_shape[0] != 1 || encoder_shape[2] != HIDDEN_DIMS {
            return Err(invalid_output(format!(
                "StyleTTS2 text encoder returned unexpected shape {encoder_shape:?}"
            )));
        }

        let decoder_tokens = Tensor::from_array((vec![1_i64, token_len], token_ids))
            .map_err(|error| {
                backend_error(format!("failed to build decoder token input: {error}"))
            })?
            .upcast();
        let decoder_hidden = Tensor::from_array((encoder_shape, encoder_values))
            .map_err(|error| {
                backend_error(format!("failed to build decoder hidden input: {error}"))
            })?
            .upcast();
        let style = Tensor::from_array((
            vec![1_i64, STYLE_VECTOR_DIMS as i64],
            self.style_vector.clone(),
        ))
        .map_err(|error| backend_error(format!("failed to build decoder style input: {error}")))?
        .upcast();
        let speed = Tensor::from_array((Vec::<i64>::new(), vec![self.speed]))
            .map_err(|error| {
                backend_error(format!("failed to build decoder speed input: {error}"))
            })?
            .upcast();

        let decoder_outputs = self
            .decoder
            .run(vec![
                ("a".to_string(), decoder_tokens),
                ("b".to_string(), decoder_hidden),
                ("c".to_string(), style),
                ("d".to_string(), speed),
            ])
            .map_err(|error| backend_error(format!("StyleTTS2 decoder failed: {error}")))?;
        let (_, samples) = extract_f32_tensor(&decoder_outputs, "z")?;
        if samples.is_empty() {
            return Err(invalid_output(
                "StyleTTS2 decoder returned an empty waveform",
            ));
        }

        Ok(StyleTts2SynthesisOutput {
            sample_rate_hz: SAMPLE_RATE_HZ,
            pcm_mono_f32: samples,
            realized_utterance: None,
        })
    }
}

fn styletts2_token_ids(sequence: &StyleTts2SymbolSequence) -> Result<Vec<i64>, StyleTts2Error> {
    let mut ipa = String::new();
    for token in &sequence.tokens {
        ipa.push_str(arpabet_to_styletts2_text(&token.symbol)?);
    }
    let ipa = ipa.trim();
    if ipa.is_empty() {
        return Ok(Vec::new());
    }

    let mut ids = Vec::with_capacity(ipa.chars().count() + 1);
    ids.push(0);
    for character in ipa.chars() {
        let id = styletts2_character_id(character).ok_or_else(|| {
            invalid_output(format!(
                "StyleTTS2 text-cleaner vocabulary has no token for `{character}`"
            ))
        })?;
        ids.push(id);
    }
    Ok(ids)
}

fn arpabet_to_styletts2_text(symbol: &str) -> Result<&'static str, StyleTts2Error> {
    let ipa = match symbol {
        "AA" => "ɑ",
        "AE" => "æ",
        "AH" => "ə",
        "AO" => "ɔ",
        "AW" => "aʊ",
        "AY" => "aɪ",
        "B" => "b",
        "CH" => "ʧ",
        "D" => "d",
        "DH" => "ð",
        "EH" => "ɛ",
        "ER" => "ɝ",
        "EY" => "eɪ",
        "F" => "f",
        "G" => "ɡ",
        "HH" => "h",
        "IH" => "ɪ",
        "IY" => "i",
        "JH" => "ʤ",
        "K" => "k",
        "L" => "l",
        "M" => "m",
        "N" => "n",
        "NG" => "ŋ",
        "OW" => "oʊ",
        "OY" => "ɔɪ",
        "P" => "p",
        "R" => "ɹ",
        "S" => "s",
        "SH" => "ʃ",
        "T" => "t",
        "TH" => "θ",
        "UH" => "ʊ",
        "UW" => "u",
        "V" => "v",
        "W" => "w",
        "Y" => "j",
        "Z" => "z",
        "ZH" => "ʒ",
        "|" => " ",
        _ => {
            return Err(invalid_output(format!(
                "cannot map lowered ARPAbet symbol `{symbol}` to StyleTTS2 text-cleaner input"
            )));
        }
    };
    Ok(ipa)
}

fn styletts2_character_id(character: char) -> Option<i64> {
    const SYMBOLS: &str = "$;:,.!?¡¿—…\"«»“” ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘̩ᵻ";
    SYMBOLS
        .chars()
        .position(|symbol| symbol == character)
        .map(|index| index as i64)
}

fn ensure_file(path: &Path, label: &str) -> Result<(), StyleTts2Error> {
    if path.is_file() {
        return Ok(());
    }
    Err(backend_error(format!(
        "{label} ONNX model file not found at {}",
        path.display()
    )))
}

fn initialize_ort_runtime() -> Result<(), StyleTts2Error> {
    static INIT: OnceLock<Result<(), String>> = OnceLock::new();
    INIT.get_or_init(initialize_ort_runtime_inner)
        .clone()
        .map_err(backend_error)
}

fn initialize_ort_runtime_inner() -> Result<(), String> {
    if let Some(path) = std::env::var_os("ORT_DYLIB_PATH").filter(|value| !value.is_empty()) {
        let path = PathBuf::from(path);
        if !path.is_file() {
            return Err(format!(
                "ORT_DYLIB_PATH points to {}, but that file does not exist",
                path.display()
            ));
        }
        return initialize_ort_runtime_from(&path);
    }

    let Some(path) = find_onnxruntime_dylib() else {
        return Err(
            "StyleTTS2 ONNX requires an ONNX Runtime shared library. Install ONNX Runtime or set ORT_DYLIB_PATH to libonnxruntime.so.".into(),
        );
    };
    initialize_ort_runtime_from(&path)
}

fn initialize_ort_runtime_from(path: &Path) -> Result<(), String> {
    ort::init_from(path)
        .map_err(|error| {
            format!(
                "failed to load ONNX Runtime dynamic library from {}: {error}",
                path.display()
            )
        })?
        .commit();
    Ok(())
}

fn find_onnxruntime_dylib() -> Option<PathBuf> {
    find_home_onnxruntime_dylib().or_else(find_linker_onnxruntime_dylib)
}

fn find_home_onnxruntime_dylib() -> Option<PathBuf> {
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let mut dirs = Vec::new();
    let local_lib = home.join(".local/lib");
    if let Ok(entries) = std::fs::read_dir(local_lib) {
        dirs.extend(entries.flatten().filter_map(|entry| {
            let file_name = entry.file_name();
            file_name
                .to_string_lossy()
                .starts_with("python")
                .then(|| entry.path().join("site-packages/onnxruntime/capi"))
        }));
    }
    find_onnxruntime_dylib_in_dirs(dirs)
}

fn find_linker_onnxruntime_dylib() -> Option<PathBuf> {
    let mut search_dirs = Vec::new();
    if let Some(paths) = std::env::var_os("LD_LIBRARY_PATH") {
        search_dirs.extend(std::env::split_paths(&paths));
    }
    search_dirs.extend([
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/usr/local/lib64"),
        PathBuf::from("/usr/lib"),
        PathBuf::from("/usr/lib64"),
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        PathBuf::from("/lib/x86_64-linux-gnu"),
    ]);
    find_onnxruntime_dylib_in_dirs(search_dirs)
}

fn find_onnxruntime_dylib_in_dirs(dirs: impl IntoIterator<Item = PathBuf>) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        candidates.extend(entries.flatten().filter_map(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            (name == "libonnxruntime.so" || name.starts_with("libonnxruntime.so."))
                .then(|| entry.path())
        }));
    }
    candidates.sort();
    candidates.pop()
}

fn load_session(path: &Path, label: &str) -> Result<Session, StyleTts2Error> {
    Session::builder()
        .map_err(|error| {
            backend_error(format!("failed to create {label} session builder: {error}"))
        })?
        .with_intra_threads(1)
        .map_err(|error| {
            backend_error(format!(
                "failed to configure {label} intra-op threads: {error}"
            ))
        })?
        .with_inter_threads(1)
        .map_err(|error| {
            backend_error(format!(
                "failed to configure {label} inter-op threads: {error}"
            ))
        })?
        .with_intra_op_spinning(false)
        .map_err(|error| {
            backend_error(format!(
                "failed to configure {label} intra-op spinning: {error}"
            ))
        })?
        .with_optimization_level(GraphOptimizationLevel::Disable)
        .map_err(|error| {
            backend_error(format!("failed to configure {label} optimization: {error}"))
        })?
        .commit_from_file(path)
        .map_err(|error| {
            backend_error(format!(
                "failed to load {label} ONNX model from {}: {error}",
                path.display()
            ))
        })
}

fn extract_f32_tensor(
    outputs: &ort::session::SessionOutputs<'_>,
    name: &str,
) -> Result<(Vec<i64>, Vec<f32>), StyleTts2Error> {
    let output = outputs
        .get(name)
        .ok_or_else(|| invalid_output(format!("StyleTTS2 inference did not return `{name}`")))?;
    let output = output
        .downcast_ref::<DynTensorValueType>()
        .map_err(|error| {
            invalid_output(format!(
                "StyleTTS2 output `{name}` is not a tensor: {error}"
            ))
        })?;
    let (shape, values) = output.try_extract_tensor::<f32>().map_err(|error| {
        invalid_output(format!("StyleTTS2 output `{name}` is not f32: {error}"))
    })?;
    if !values.iter().all(|value| value.is_finite()) {
        return Err(invalid_output(format!(
            "StyleTTS2 output `{name}` contains non-finite values"
        )));
    }
    Ok((shape.to_vec(), values.to_vec()))
}

fn backend_error(message: impl Into<String>) -> StyleTts2Error {
    StyleTts2Error::Backend {
        message: message.into(),
    }
}

fn invalid_output(reason: impl Into<String>) -> StyleTts2Error {
    StyleTts2Error::InvalidOutput {
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::symbols::StyleTts2SymbolSource;
    use crate::symbols::StyleTts2SymbolToken;

    #[test]
    fn arpabet_symbols_map_to_styletts2_token_ids() {
        let ids = styletts2_token_ids(&StyleTts2SymbolSequence {
            tokens: vec![
                token("HH"),
                token("AH"),
                token("L"),
                token("OW"),
                token("|"),
                token("W"),
                token("ER"),
                token("L"),
                token("D"),
            ],
        })
        .expect("token ids");

        assert_eq!(ids[0], 0);
        assert!(ids.iter().all(|id| (0..178).contains(id)));
        assert!(ids.len() > 9);
    }

    fn token(symbol: &str) -> StyleTts2SymbolToken {
        StyleTts2SymbolToken {
            symbol: symbol.into(),
            source: StyleTts2SymbolSource::Phoneme,
        }
    }
}
