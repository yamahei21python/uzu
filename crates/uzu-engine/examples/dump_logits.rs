//! Dump teacher-forced logits for a token sequence.
//!
//! Usage:
//!   cargo run -p uzu-engine --example dump_logits -- \
//!     <model-dir> <output.bin> <token_id,token_id,...>
//!
//! Output: raw little-endian f32, shape [len(tokens), vocab].

use std::{env, fs, path::PathBuf};

use uzu_engine::{
    backends::{Backend, BackendSelection, select_backend},
    engine::Engine,
};

struct Dump {
    model: PathBuf,
    tokens: Vec<u64>,
    output: PathBuf,
}

impl BackendSelection for Dump {
    type Output = usize;
    type Error = String;

    async fn select<B: Backend>(self) -> Result<usize, String> {
        let engine = Engine::<B>::new().await.map_err(|error| error.to_string())?;
        let model = engine
            .load_language_model(&self.model)
            .await
            .map_err(|error| error.to_string())?;
        let mut state = model
            .create_empty_state(None, 0)
            .map_err(|error| error.to_string())?;

        let mut logits = Vec::new();
        for chunk in self.tokens.chunks(1024) {
            logits.extend(
                model
                    .teacher_forced_logits(chunk, &mut state)
                    .await
                    .map_err(|error| error.to_string())?,
            );
        }

        let bytes = logits.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<u8>>();
        fs::write(&self.output, bytes).map_err(|error| error.to_string())?;
        Ok(self.tokens.len())
    }
}

fn parse_args() -> Result<Dump, String> {
    let mut args = env::args().skip(1);
    let model = PathBuf::from(args.next().ok_or("missing model dir")?);
    let output = PathBuf::from(args.next().ok_or("missing output path")?);
    let tokens = args
        .next()
        .ok_or("missing comma-separated token ids")?
        .split(',')
        .map(|value| value.trim().parse::<u64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("invalid token id: {error}"))?;
    if tokens.is_empty() {
        return Err("empty token list".to_owned());
    }
    Ok(Dump { model, tokens, output })
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let dump = parse_args()?;
    let rows = select_backend(dump, "unknown backend".to_owned()).await?;
    eprintln!("dumped {rows} rows");
    Ok(())
}
