use mesh_client::network::router::{classify, strip_split_suffix, Category, Complexity};
use serde_json::json;

#[test]
fn router_classifies_code_request() {
    let body = json!({
        "messages": [{"role": "user", "content": "Write a Python function to implement binary search and debug any issues"}]
    });
    let cl = classify(&body);
    assert_eq!(cl.category, Category::Code);
}

#[test]
fn router_classifies_chat_default() {
    let body = json!({
        "messages": [{"role": "user", "content": "Hi"}]
    });
    let cl = classify(&body);
    assert_eq!(cl.category, Category::Chat);
    assert!(matches!(cl.complexity, Complexity::Quick));
}

#[test]
fn router_strip_split_suffix_works() {
    assert_eq!(
        strip_split_suffix("MiniMax-M2.5-Q4_K_M-00001-of-00004"),
        "MiniMax-M2.5-Q4_K_M"
    );
    assert_eq!(strip_split_suffix("SomeModel"), "SomeModel");
    assert_eq!(strip_split_suffix(""), "");
}
