# MeshLLM Swift SDK

Swift Package for connecting to mesh-llm meshes from iOS, Mac Catalyst, and macOS apps.

## Installation

Add to your app's `Package.swift` using a tagged release:

```swift
dependencies: [
    .package(url: "https://github.com/Mesh-LLM/mesh-llm", from: "0.1.0"),
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "MeshLLM", package: "mesh-llm"),
        ]
    ),
]
```

The repo root is the Swift package entrypoint. Tagged releases resolve the
prebuilt XCFramework automatically through SwiftPM.

For development from a local checkout, build the native artifact first:

```bash
./sdk/swift/scripts/build-xcframework.sh
```

That generates `sdk/swift/Generated/MeshLLMFFI.xcframework`, which the root
Swift package picks up automatically for the real UniFFI-backed implementation.

If you only want to run the pure Swift fallback without the XCFramework, set:

```bash
MESH_SWIFT_FORCE_STUB=1 swift test
```

## Usage

```swift
import MeshLLM

let publicMeshes = try await MeshClient.discoverPublicMeshes()
let client = MeshClient(
    inviteToken: publicMeshes[0].inviteToken,
    ownerKeypairBytesHex: loadPersistedOwnerKeypair()
)
try await client.join()

let models = try await client.listModels()
let request = ChatRequest(model: models[0].id, messages: [
    ChatMessage(role: "user", content: "Hello!")
])

for try await event in client.chatStream(request) {
    switch event {
    case .tokenDelta(_, let delta):
        print(delta, terminator: "")
    case .completed:
        print()
    default:
        break
    }
}
```

Or let the SDK pick the best public mesh with smart auto:

```swift
let client = try await MeshClient.connectPublic(
    ownerKeypairBytesHex: loadPersistedOwnerKeypair()
)
try await client.join()
```

## App Store Export Compliance

### Encryption

mesh-llm uses QUIC (via iroh) for transport, which uses TLS 1.3. This constitutes use of encryption.

**Required**: Set `ITSAppUsesNonExemptEncryption = YES` in your app's `Info.plist`.

If your app qualifies for an exemption (e.g., uses only standard encryption), you may set `ITSAppUsesNonExemptEncryption = NO` and provide justification.

### Privacy Manifest

The MeshLLM XCFramework includes a `PrivacyInfo.xcprivacy` manifest declaring:
- `NSPrivacyTracking = false` (no tracking)
- No data collection
- No required-reason API usage

This manifest is embedded inside each `.framework` bundle in the XCFramework, satisfying Apple's requirement since Spring 2024.

### Entitlements

No special entitlements are required. mesh-llm uses standard POSIX sockets via iroh/quinn — no `com.apple.security.network.client` entitlement is needed for macOS (it's allowed by default).

For iOS, network access is allowed by default. No special entitlements needed.

### App Store Submission Checklist

- [ ] Set `ITSAppUsesNonExemptEncryption` in `Info.plist`
- [ ] Verify `PrivacyInfo.xcprivacy` is embedded in XCFramework (run `find MeshLLM.xcframework -name PrivacyInfo.xcprivacy`)
- [ ] No subprocess spawning (mesh-llm SDK never calls `Process()`)
- [ ] No filesystem access for credentials (pass keys via constructor)
- [ ] Implement `reconnect()` in `UIApplication.willEnterForegroundNotification` observer

## iOS Backgrounding

Register for foreground notifications to reconnect after backgrounding:

```swift
NotificationCenter.default.addObserver(
    forName: UIApplication.willEnterForegroundNotification,
    object: nil,
    queue: .main
) { _ in
    Task {
        try? await client.reconnect()
    }
}
```
