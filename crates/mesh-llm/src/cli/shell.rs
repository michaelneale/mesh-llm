pub(crate) fn single_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_quote_wraps_and_escapes_embedded_quotes() {
        assert_eq!(single_quote("Qwen 3.6 27B"), "'Qwen 3.6 27B'");
        assert_eq!(single_quote("Qwen's model"), "'Qwen'\"'\"'s model'");
    }
}
