use anyhow::Result;
use mlx_rs::Array;

#[derive(Clone, Debug, Default)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub seed: Option<u64>,
}

pub struct Sampler {
    _params: SamplingParams,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        Self { _params: params }
    }

    pub fn sample_next_token(&mut self, logits: &Array) -> Result<u32> {
        super::model::argmax_last(logits)
    }
}

pub struct StopBuffer {
    stops: Vec<String>,
    buffer: String,
}

pub struct StopChunk {
    pub emit: String,
    pub matched: bool,
}

impl StopBuffer {
    pub fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            buffer: String::new(),
        }
    }

    pub fn push(&mut self, text: &str) -> StopChunk {
        self.buffer.push_str(text);
        for stop in &self.stops {
            if let Some(index) = self.buffer.find(stop) {
                let emit = self.buffer[..index].to_string();
                self.buffer.clear();
                return StopChunk {
                    emit,
                    matched: true,
                };
            }
        }

        let keep = self.max_partial_suffix();
        let split_at = self.buffer.len().saturating_sub(keep);
        let emit = self.buffer[..split_at].to_string();
        self.buffer.drain(..split_at);
        StopChunk {
            emit,
            matched: false,
        }
    }

    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.buffer)
    }

    fn max_partial_suffix(&self) -> usize {
        let mut best = 0;
        for stop in &self.stops {
            let max_len = stop.len().saturating_sub(1);
            for len in 1..=max_len {
                if self.buffer.ends_with(&stop[..len]) {
                    best = best.max(len);
                }
            }
        }
        best
    }
}
