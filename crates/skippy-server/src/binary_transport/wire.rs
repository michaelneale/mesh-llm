use std::{io, thread, time::Duration};

use anyhow::{bail, Result};
use skippy_protocol::binary::{write_stage_message, StageWireMessage, WireActivationDType};

#[derive(Clone, Copy, Debug)]
pub struct WireCondition {
    delay_ms: f64,
    mbps: Option<f64>,
}

impl WireCondition {
    pub fn new(delay_ms: f64, mbps: Option<f64>) -> Result<Self> {
        if delay_ms < 0.0 {
            bail!("downstream wire delay must not be negative");
        }
        if mbps.is_some_and(|value| value <= 0.0) {
            bail!("downstream wire mbps must be greater than zero");
        }
        Ok(Self { delay_ms, mbps })
    }

    fn sleep_for(&self, message: &StageWireMessage) {
        let delay_seconds = self.delay_ms / 1000.0;
        let bandwidth_seconds = self
            .mbps
            .map(|mbps| conditioned_wire_bytes(message) as f64 / (mbps * 125_000.0))
            .unwrap_or(0.0);
        let seconds = delay_seconds + bandwidth_seconds;
        if seconds > 0.0 {
            thread::sleep(Duration::from_secs_f64(seconds));
        }
    }
}

fn conditioned_wire_bytes(message: &StageWireMessage) -> usize {
    let header_bytes: usize = 4 * 4 + 9 * 4;
    let token_bytes = message
        .tokens
        .len()
        .saturating_mul(std::mem::size_of::<i32>());
    header_bytes
        .saturating_add(token_bytes)
        .saturating_add(message.activation.len())
        .saturating_add(message.raw_bytes.len())
}

pub(crate) fn write_stage_message_conditioned(
    writer: impl io::Write,
    message: &StageWireMessage,
    dtype: WireActivationDType,
    condition: WireCondition,
) -> io::Result<()> {
    condition.sleep_for(message);
    write_stage_message(writer, message, dtype)
}
