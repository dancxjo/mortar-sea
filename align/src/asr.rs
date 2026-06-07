use std::{
    io::{BufRead, BufReader, Write},
    path::Path,
    process::{Command, Stdio},
};

use anyhow::Context;

use crate::{AsrSentence, Ear, EarRequest, EarResponse};

pub(crate) fn transcribe_with_ear(
    samples: Vec<f32>,
    duration_ms: u64,
) -> anyhow::Result<Vec<AsrSentence>> {
    let model_path = mortar_sea::models::ensure_asr_whisper_model_available()?;
    let mut ear = Ear::spawn(&model_path)?;
    ear.transcribe(samples, duration_ms)
}

impl Ear {
    fn spawn(model_path: &Path) -> anyhow::Result<Self> {
        let worker =
            std::env::var_os("MORTAR_EAR").or_else(|| std::env::var_os("MORTAR_ASR_WORKER"));
        let mut command = if let Some(worker) = worker {
            let mut command = Command::new(worker);
            command.arg(model_path);
            command
        } else {
            let mut command =
                Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string()));
            command.args(["run", "-q", "-p", "ear", "--"]);
            command.arg(model_path);
            command
        };
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .context("failed to spawn ear")?;
        let stdin = child.stdin.take().context("ear stdin unavailable")?;
        let stdout = child.stdout.take().context("ear stdout unavailable")?;
        Ok(Self {
            _child: child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 0,
        })
    }

    fn transcribe(
        &mut self,
        samples: Vec<f32>,
        duration_ms: u64,
    ) -> anyhow::Result<Vec<AsrSentence>> {
        self.next_id = self.next_id.checked_add(1).context("ear id overflow")?;
        let id = self.next_id;
        serde_json::to_writer(
            &mut self.stdin,
            &EarRequest {
                id,
                samples,
                duration_ms,
            },
        )?;
        writeln!(self.stdin)?;
        self.stdin.flush()?;

        let mut line = String::new();
        loop {
            line.clear();
            let read = self.stdout.read_line(&mut line)?;
            anyhow::ensure!(read > 0, "ear exited before response");
            let response = serde_json::from_str::<EarResponse>(&line)?;
            if response.id != id {
                continue;
            }
            if let Some(error) = response.error {
                anyhow::bail!("ear failed: {error}");
            }
            return Ok(response.sentences);
        }
    }
}
