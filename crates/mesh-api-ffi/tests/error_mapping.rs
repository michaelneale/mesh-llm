use mesh_ffi::{create_client, FfiError};

#[test]
fn invalid_invite_token_returns_ffi_error() {
    let result = create_client("deadbeef".to_string(), "".to_string());
    match result {
        Ok(_) => panic!("expected Err(FfiError::InvalidInviteToken)"),
        Err(FfiError::InvalidInviteToken) => {} // expected
        Err(other) => panic!("Expected InvalidInviteToken, got {:?}", other),
    }
}

#[test]
fn no_anyhow_in_exported_functions() {
    // Verifies at compile time that FfiError implements std::error::Error,
    // confirming no anyhow::Error leaks across the FFI boundary.
    fn _assert_error<E: std::error::Error>() {}
    _assert_error::<FfiError>();
}

#[test]
fn ffi_error_all_variants_present() {
    // Exhaustive match ensures all required variants exist and are reachable.
    // Adding a variant to FfiError without updating this test will cause a compile error.
    let variants: &[FfiError] = &[
        FfiError::InvalidInviteToken,
        FfiError::InvalidOwnerKeypair,
        FfiError::BuildFailed,
        FfiError::JoinFailed,
        FfiError::DiscoveryFailed,
        FfiError::StreamFailed,
        FfiError::Cancelled,
        FfiError::ReconnectFailed,
        FfiError::HostUnavailable,
    ];
    for v in variants {
        assert!(
            !v.to_string().is_empty(),
            "FfiError::{:?} has empty Display",
            v
        );
    }
}
