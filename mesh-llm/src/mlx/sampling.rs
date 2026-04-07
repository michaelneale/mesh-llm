use anyhow::Result;
use mlx_rs::Array;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: None,
            seed: None,
        }
    }
}

pub struct Sampler {
    params: SamplingParams,
    rng: Option<StdRng>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let rng = params.seed.map(StdRng::seed_from_u64);
        Self { params, rng }
    }

    pub fn sample_next_token(&mut self, logits: &Array) -> Result<u32> {
        if self.params.temperature <= 0.0 {
            return crate::mlx::model::argmax_last(logits);
        }

        let mut candidates = last_logits(logits)?
            .into_iter()
            .enumerate()
            .map(|(token, logit)| (token as u32, logit / self.params.temperature))
            .collect::<Vec<_>>();

        candidates.sort_by(|left, right| right.1.total_cmp(&left.1));

        if let Some(top_k) = self.params.top_k {
            let top_k = top_k.max(1).min(candidates.len());
            candidates.truncate(top_k);
        }

        let max_logit = candidates
            .iter()
            .map(|(_, logit)| *logit)
            .max_by(|left, right| left.total_cmp(right))
            .unwrap_or(0.0);

        let mut weighted = candidates
            .into_iter()
            .map(|(token, logit)| (token, (logit - max_logit).exp()))
            .collect::<Vec<_>>();
        let mut total = weighted.iter().map(|(_, weight)| *weight).sum::<f32>();

        if self.params.top_p < 1.0 {
            let mut cumulative = 0.0f32;
            let mut kept = 0usize;
            for (_, weight) in &weighted {
                cumulative += *weight / total.max(f32::EPSILON);
                kept += 1;
                if cumulative >= self.params.top_p.max(0.0) {
                    break;
                }
            }
            weighted.truncate(kept.max(1));
            total = weighted.iter().map(|(_, weight)| *weight).sum::<f32>();
        }

        let mut draw = self.random_f32() * total.max(f32::EPSILON);
        for (token, weight) in weighted {
            draw -= weight;
            if draw <= 0.0 {
                return Ok(token);
            }
        }

        Ok(crate::mlx::model::argmax_last(logits)?)
    }

    fn random_f32(&mut self) -> f32 {
        if let Some(rng) = &mut self.rng {
            rng.random::<f32>()
        } else {
            rand::random::<f32>()
        }
    }
}

pub fn top_logits(logits: &Array, limit: usize) -> Result<Vec<(u32, f32)>> {
    let mut candidates = last_logits(logits)?
        .into_iter()
        .enumerate()
        .map(|(token, logit)| (token as u32, logit))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    candidates.truncate(limit.max(1).min(candidates.len()));
    Ok(candidates)
}

pub struct StopBuffer {
    sequences: Vec<String>,
    pending: String,
    holdback_chars: usize,
    matched: bool,
}

impl StopBuffer {
    pub fn new(sequences: Vec<String>) -> Self {
        let holdback_chars = sequences
            .iter()
            .map(|sequence| sequence.chars().count().saturating_sub(1))
            .max()
            .unwrap_or(0);
        Self {
            sequences,
            pending: String::new(),
            holdback_chars,
            matched: false,
        }
    }

    pub fn push(&mut self, text: &str) -> StopChunk {
        if self.matched {
            return StopChunk::default();
        }

        self.pending.push_str(text);

        if self.sequences.is_empty() {
            return StopChunk {
                emit: std::mem::take(&mut self.pending),
                matched: false,
            };
        }

        if let Some(index) = find_earliest_stop(&self.pending, &self.sequences) {
            let emit = self.pending[..index].to_string();
            self.pending.clear();
            self.matched = true;
            return StopChunk {
                emit,
                matched: true,
            };
        }

        let safe_len = safe_prefix_len(&self.pending, self.holdback_chars);
        if safe_len == 0 {
            return StopChunk::default();
        }

        let emit = self.pending[..safe_len].to_string();
        self.pending.drain(..safe_len);
        StopChunk {
            emit,
            matched: false,
        }
    }

    pub fn finish(&mut self) -> String {
        if self.matched {
            String::new()
        } else {
            std::mem::take(&mut self.pending)
        }
    }
}

#[derive(Default)]
pub struct StopChunk {
    pub emit: String,
    pub matched: bool,
}

fn last_logits(logits: &Array) -> Result<Vec<f32>> {
    let shape = logits.shape();
    let flat = if shape.len() == 3 {
        let last_idx = (shape[1] - 1) as i32;
        let idx = Array::from_int(last_idx);
        logits.take_axis(&idx, 1)?.reshape(&[-1])?
    } else {
        logits.reshape(&[-1])?
    };
    let flat = flat.as_type::<f32>()?;
    mlx_rs::transforms::eval([&flat])?;
    Ok(flat.as_slice::<f32>().to_vec())
}

fn find_earliest_stop(text: &str, sequences: &[String]) -> Option<usize> {
    sequences
        .iter()
        .filter_map(|sequence| text.find(sequence))
        .min()
}

fn safe_prefix_len(text: &str, holdback_chars: usize) -> usize {
    if holdback_chars == 0 {
        return text.len();
    }
    let total_chars = text.chars().count();
    if total_chars <= holdback_chars {
        return 0;
    }
    let safe_chars = total_chars - holdback_chars;
    text.char_indices()
        .nth(safe_chars)
        .map(|(index, _)| index)
        .unwrap_or(text.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_buffer_holds_back_partial_match() {
        let mut buffer = StopBuffer::new(vec!["</s>".to_string()]);
        let first = buffer.push("hello</");
        assert_eq!(first.emit, "hell");
        assert!(!first.matched);

        let second = buffer.push("s>world");
        assert_eq!(second.emit, "o");
        assert!(second.matched);
        assert!(buffer.finish().is_empty());
    }

    #[test]
    fn stop_buffer_flushes_when_no_stop_matches() {
        let mut buffer = StopBuffer::new(vec!["STOP".to_string()]);
        let first = buffer.push("hel");
        assert!(first.emit.is_empty());
        let second = buffer.push("lo");
        assert_eq!(second.emit, "he");
        assert_eq!(buffer.finish(), "llo");
    }
}
