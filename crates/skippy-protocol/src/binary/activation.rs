use std::io;

use super::{invalid_data, WireActivationDType};

pub fn activation_wire_bytes(
    dtype: WireActivationDType,
    token_count: i32,
    n_embd: i32,
) -> io::Result<usize> {
    if token_count < 0 || n_embd < 0 {
        return Err(invalid_data("negative activation dimensions"));
    }
    let token_count = token_count as usize;
    let n_embd = n_embd as usize;
    let elements = token_count
        .checked_mul(n_embd)
        .ok_or_else(|| invalid_data("activation element count overflow"))?;
    match dtype {
        WireActivationDType::F32 => elements
            .checked_mul(4)
            .ok_or_else(|| invalid_data("activation byte count overflow")),
        WireActivationDType::F16 => elements
            .checked_mul(2)
            .ok_or_else(|| invalid_data("activation byte count overflow")),
        WireActivationDType::Q8 => token_count
            .checked_mul(4)
            .and_then(|scales| scales.checked_add(elements))
            .ok_or_else(|| invalid_data("activation byte count overflow")),
    }
}

pub fn encode_f32_activation_payload(
    dtype: WireActivationDType,
    token_count: i32,
    n_embd: i32,
    f32_payload: &[u8],
) -> io::Result<Vec<u8>> {
    let expected_f32_bytes = activation_wire_bytes(WireActivationDType::F32, token_count, n_embd)?;
    if f32_payload.len() != expected_f32_bytes {
        return Err(invalid_data("F32 activation payload size mismatch"));
    }
    match dtype {
        WireActivationDType::F32 => Ok(f32_payload.to_vec()),
        WireActivationDType::F16 => encode_f32_to_f16_bytes(f32_payload),
        WireActivationDType::Q8 => encode_f32_to_q8_bytes(f32_payload, token_count, n_embd),
    }
}

pub(crate) fn decode_f16_to_f32_bytes(input: &[u8]) -> io::Result<Vec<u8>> {
    if !input.len().is_multiple_of(2) {
        return Err(invalid_data("F16 activation payload has odd byte length"));
    }
    let mut out = Vec::with_capacity((input.len() / 2) * 4);
    for chunk in input.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.extend_from_slice(&f16_bits_to_f32(bits).to_le_bytes());
    }
    Ok(out)
}

fn encode_f32_to_f16_bytes(input: &[u8]) -> io::Result<Vec<u8>> {
    if !input.len().is_multiple_of(4) {
        return Err(invalid_data("F32 activation payload size is not aligned"));
    }
    let mut out = Vec::with_capacity(input.len() / 2);
    for chunk in input.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().expect("chunks_exact size"));
        out.extend_from_slice(&f32_to_f16_bits(value).to_le_bytes());
    }
    Ok(out)
}

pub(crate) fn decode_q8_to_f32_bytes(
    input: &[u8],
    token_count: i32,
    n_embd: i32,
) -> io::Result<Vec<u8>> {
    if token_count < 0 || n_embd < 0 {
        return Err(invalid_data("negative Q8 activation dimensions"));
    }
    let token_count = token_count as usize;
    let n_embd = n_embd as usize;
    let scale_bytes = token_count
        .checked_mul(4)
        .ok_or_else(|| invalid_data("Q8 scale byte count overflow"))?;
    let value_bytes = token_count
        .checked_mul(n_embd)
        .ok_or_else(|| invalid_data("Q8 value byte count overflow"))?;
    if input.len() != scale_bytes + value_bytes {
        return Err(invalid_data("Q8 activation payload size mismatch"));
    }
    let mut out = Vec::with_capacity(value_bytes * 4);
    for token_index in 0..token_count {
        let scale_offset = token_index * 4;
        let scale = f32::from_le_bytes([
            input[scale_offset],
            input[scale_offset + 1],
            input[scale_offset + 2],
            input[scale_offset + 3],
        ]);
        let row_offset = scale_bytes + token_index * n_embd;
        for value in &input[row_offset..row_offset + n_embd] {
            let signed = *value as i8;
            out.extend_from_slice(&((signed as f32) * scale).to_le_bytes());
        }
    }
    Ok(out)
}

fn encode_f32_to_q8_bytes(input: &[u8], token_count: i32, n_embd: i32) -> io::Result<Vec<u8>> {
    if token_count < 0 || n_embd < 0 {
        return Err(invalid_data("negative Q8 activation dimensions"));
    }
    let token_count = token_count as usize;
    let n_embd = n_embd as usize;
    let expected_bytes = token_count
        .checked_mul(n_embd)
        .and_then(|elements| elements.checked_mul(4))
        .ok_or_else(|| invalid_data("Q8 source byte count overflow"))?;
    if input.len() != expected_bytes {
        return Err(invalid_data("Q8 source payload size mismatch"));
    }

    let mut scales = Vec::with_capacity(token_count * 4);
    let mut packed = Vec::with_capacity(token_count * n_embd);
    for token_index in 0..token_count {
        let row_offset = token_index * n_embd * 4;
        let row = &input[row_offset..row_offset + n_embd * 4];
        let mut max_abs = 0.0_f32;
        for chunk in row.chunks_exact(4) {
            let value = f32::from_le_bytes(chunk.try_into().expect("chunks_exact size"));
            max_abs = max_abs.max(value.abs());
        }
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        scales.extend_from_slice(&scale.to_le_bytes());
        for chunk in row.chunks_exact(4) {
            let value = f32::from_le_bytes(chunk.try_into().expect("chunks_exact size"));
            let quantized = (value / scale).round().clamp(-127.0, 127.0) as i8;
            packed.push(quantized as u8);
        }
    }
    scales.extend_from_slice(&packed);
    Ok(scales)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x03ff;
    let f32_bits = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut mant = mantissa as u32;
            let mut exp = -14_i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x03ff;
            let exp_bits = ((exp + 127) as u32) << 23;
            sign | exp_bits | (mant << 13)
        }
    } else if exponent == 0x1f {
        sign | 0x7f80_0000 | ((mantissa as u32) << 13)
    } else {
        let exp_bits = ((exponent as u32) + (127 - 15)) << 23;
        sign | exp_bits | ((mantissa as u32) << 13)
    };
    f32::from_bits(f32_bits)
}

fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exponent == 0xff {
        let nan_bit = if mantissa == 0 { 0 } else { 0x0200 };
        return sign | 0x7c00 | nan_bit;
    }

    let half_exp = exponent - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mant = mantissa | 0x0080_0000;
        let shift = (14 - half_exp) as u32;
        let mut half_mant = (mant >> shift) as u16;
        if ((mant >> (shift - 1)) & 1) != 0 {
            half_mant = half_mant.saturating_add(1);
        }
        return sign | half_mant;
    }

    let mut half = sign | ((half_exp as u16) << 10) | ((mantissa >> 13) as u16);
    if (mantissa & 0x0000_1000) != 0 {
        half = half.saturating_add(1);
    }
    half
}
