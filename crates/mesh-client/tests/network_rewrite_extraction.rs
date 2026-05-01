use mesh_client::network::rewrite::{new_rewrite_map, PortRewriteMap};

#[test]
fn rewrite_map_creation_and_clone() {
    let map: PortRewriteMap = new_rewrite_map();
    let _map2 = map.clone();
    let guard = map.blocking_read();
    assert!(guard.is_empty());
}

#[test]
fn rewrite_map_port_insert_and_read() {
    let map = new_rewrite_map();
    {
        let mut guard = map.blocking_write();
        guard.insert(49502, 50001);
    }
    let guard = map.blocking_read();
    assert_eq!(guard.get(&49502), Some(&50001));
    assert_eq!(guard.get(&9999), None);
}
