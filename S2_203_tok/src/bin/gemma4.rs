use std::io::{self, Write};
use std::path::Path;
use std::sync::mpsc;
use std::time::Instant;

use furiosa_opt_gemma4::host::generate::{self, Delta, GenerationRequest};
use furiosa_opt_gemma4::host::runtime::Workspace;
use furiosa_opt_gemma4::host::sampling::SamplingConfig;
use furiosa_opt_gemma4::host::tokenizer::Tokenizer;
use furiosa_opt_gemma4::host::{audio, image, load};
use furiosa_opt_std::prelude::*;

const DEFAULT_PROMPT: &str = "Write a hello world program in Rust.";

struct Args {
    image_path: Option<String>,
    audio_path: Option<String>,
    /// Keep only the first N soft tokens (image patches or audio frames): short multimodal tests.
    max_media: Option<usize>,
    prompt: String,
}

fn parse_args() -> Result<Args, Box<dyn std::error::Error>> {
    let mut image_path = None;
    let mut audio_path = None;
    let mut max_media = None;
    let mut words = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--image" {
            image_path = Some(args.next().ok_or("--image requires a path")?);
        } else if arg == "--audio" {
            audio_path = Some(args.next().ok_or("--audio requires a path")?);
        } else if arg == "--max-media" {
            max_media = Some(args.next().ok_or("--max-media requires a count")?.parse::<usize>()?);
        } else {
            words.push(arg);
        }
    }
    let prompt = if words.is_empty() {
        DEFAULT_PROMPT.to_owned()
    } else {
        words.join(" ")
    };
    Ok(Args { image_path, audio_path, max_media, prompt })
}

fn build_messages(prompt: &str, has_image: bool, has_audio: bool) -> Vec<serde_json::Value> {
    let content = if has_audio {
        serde_json::json!([{ "type": "text", "text": prompt }, { "type": "audio" }])
    } else if has_image {
        serde_json::json!([{ "type": "image" }, { "type": "text", "text": prompt }])
    } else {
        serde_json::json!(prompt)
    };
    vec![serde_json::json!({ "role": "user", "content": content })]
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args()?;
    let max_new = std::env::var("GEMMA4_MAX_NEW_TOKENS")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?
        .unwrap_or(generate::DEFAULT_MAX_NEW_TOKENS);

    let mut device = Device::new(furiosa_opt_gemma4::ops::embed_token.topology())?;
    let load_start = Instant::now();
    let weights = load::load_weights(&mut device).await?;
    eprintln!("model loaded in {:.2?}", load_start.elapsed());

    let mut image = args
        .image_path
        .map(|path| image::load_from_bytes(&std::fs::read(Path::new(&path))?))
        .transpose()?;
    let mut audio = args
        .audio_path
        .map(|path| audio::load_from_bytes(&std::fs::read(Path::new(&path))?, "wav"))
        .transpose()?;
    if let Some(keep) = args.max_media {
        if let Some(patches) = &mut image {
            patches.pixels.truncate(keep);
            patches.positions.truncate(keep);
        }
        if let Some(frames) = &mut audio {
            frames.frames.truncate(keep);
        }
    }
    if let Some(frames) = &audio {
        eprintln!("audio: {} frames", frames.len());
    }
    if let Some(patches) = &image {
        eprintln!("image: {} patches", patches.pixels.len());
    }

    let tokenizer = Tokenizer::new(&load::tokenizer_path(&weights))?;
    let messages = build_messages(&args.prompt, image.is_some(), audio.is_some());
    let chat_ids = tokenizer.encode_chat(&messages, false)?;
    let (prompt_ids, soft_at) = generate::splice_multimodal_tokens(chat_ids, image.as_ref(), audio.as_ref())?;

    let mut workspace = Workspace::new(&mut device, &weights, prompt_ids.len().saturating_add(max_new), 1).await?;
    let mut model = furiosa_opt_gemma4::host::model::Model::new(&mut device, weights, &mut workspace).await?;

    println!("{}", args.prompt);
    io::stdout().flush()?;

    let req = GenerationRequest {
        prompt_ids,
        soft_at,
        image,
        audio,
        max_new_tokens: max_new,
        sampling: SamplingConfig::DEFAULT,
        stop_strings: Vec::new(),
        separate_reasoning: false,
        ignore_eos: false,
    };

    let (print_tx, print_rx) = mpsc::channel::<String>();
    let printer = std::thread::spawn(move || {
        for text in print_rx {
            print!("{text}");
            let _ = io::stdout().flush();
        }
    });

    let run_start = Instant::now();
    let mut first_token_at = None;
    let output = generate::generate(&mut device, &mut model, &tokenizer, &mut workspace, req, |delta| {
        if first_token_at.is_none() {
            first_token_at = Some(Instant::now());
        }
        if let Delta::Content(text) = delta {
            let _ = print_tx.send(text.to_owned());
        }
        true
    })
    .await?;
    drop(print_tx);
    let _ = printer.join();
    println!();

    let ttft = first_token_at.unwrap_or_else(Instant::now) - run_start;
    let decode_elapsed = run_start.elapsed().saturating_sub(ttft);
    let prefill_tok_per_s = output.prompt_tokens as f64 / ttft.as_secs_f64();
    let decode_tok_per_s = if decode_elapsed.as_secs_f64() > 0.0 {
        output.completion_tokens as f64 / decode_elapsed.as_secs_f64()
    } else {
        0.0
    };
    eprintln!(
        "ttft: {ttft:.2?} ({} prompt tokens, {prefill_tok_per_s:.2} tok/s prefill) | decode: {} tokens in {decode_elapsed:.2?} ({decode_tok_per_s:.2} tok/s) | finish: {:?}",
        output.prompt_tokens, output.completion_tokens, output.finish_reason,
    );
    Ok(())
}
