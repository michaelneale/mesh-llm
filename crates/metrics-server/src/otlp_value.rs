use opentelemetry_proto::tonic::common::v1::{any_value, AnyValue, KeyValue};
use serde_json::{json, Map, Value};

pub(crate) fn attributes_to_json(attributes: &[KeyValue]) -> Value {
    let mut map = Map::new();
    for attribute in attributes {
        map.insert(
            attribute.key.clone(),
            attribute
                .value
                .as_ref()
                .map(any_value_to_json)
                .unwrap_or(Value::Null),
        );
    }
    Value::Object(map)
}

pub(crate) fn empty_string_to_none(value: &str) -> Option<&str> {
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

pub(crate) fn any_value_to_json(value: &AnyValue) -> Value {
    match value.value.as_ref() {
        Some(any_value::Value::StringValue(value)) => Value::String(value.clone()),
        Some(any_value::Value::BoolValue(value)) => Value::Bool(*value),
        Some(any_value::Value::IntValue(value)) => json!(value),
        Some(any_value::Value::DoubleValue(value)) => json!(value),
        Some(any_value::Value::ArrayValue(value)) => {
            Value::Array(value.values.iter().map(any_value_to_json).collect())
        }
        Some(any_value::Value::KvlistValue(value)) => attributes_to_json(&value.values),
        Some(any_value::Value::BytesValue(value)) => Value::String(bytes_to_hex(value)),
        None => Value::Null,
    }
}

pub(crate) fn attribute_string(attributes: &[KeyValue], key: &str) -> Option<String> {
    attributes
        .iter()
        .find(|attribute| attribute.key == key)
        .and_then(|attribute| attribute.value.as_ref())
        .and_then(any_value_to_string)
}

pub(crate) fn attribute_string_from_value(attributes: &Value, key: &str) -> Option<String> {
    attributes.get(key).and_then(|value| match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    })
}

pub(crate) fn any_value_to_string(value: &AnyValue) -> Option<String> {
    match value.value.as_ref()? {
        any_value::Value::StringValue(value) => Some(value.clone()),
        any_value::Value::BoolValue(value) => Some(value.to_string()),
        any_value::Value::IntValue(value) => Some(value.to_string()),
        any_value::Value::DoubleValue(value) => Some(value.to_string()),
        any_value::Value::BytesValue(value) => Some(bytes_to_hex(value)),
        any_value::Value::ArrayValue(_) | any_value::Value::KvlistValue(_) => None,
    }
}

pub(crate) fn kv_string(key: &str, value: &str) -> KeyValue {
    KeyValue {
        key: key.to_string(),
        value: Some(AnyValue {
            value: Some(any_value::Value::StringValue(value.to_string())),
        }),
    }
}

pub(crate) fn kv_i64(key: &str, value: i64) -> KeyValue {
    KeyValue {
        key: key.to_string(),
        value: Some(AnyValue {
            value: Some(any_value::Value::IntValue(value)),
        }),
    }
}

pub(crate) fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        let _ = write!(out, "{byte:02x}");
    }
    out
}
