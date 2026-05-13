use super::*;

pub(super) fn trim_at_stop<'a>(text: &'a str, stop_values: &[&str]) -> &'a str {
    let first_stop = stop_values
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min();
    match first_stop {
        Some(index) => &text[..index],
        None => text,
    }
}

pub(super) fn valid_utf8_prefix_len(bytes: &[u8]) -> usize {
    match std::str::from_utf8(bytes) {
        Ok(_) => bytes.len(),
        Err(error) => error.valid_up_to(),
    }
}

pub(super) fn saturating_u32(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

pub(super) fn now_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
}

pub(super) fn stable_wire_id(parts: &[&[u8]]) -> u64 {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let id = u64::from_le_bytes(
        digest[..8]
            .try_into()
            .expect("sha256 digest has an 8-byte prefix"),
    );
    if id == 0 {
        1
    } else {
        id
    }
}
