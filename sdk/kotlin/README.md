# MeshLLM Kotlin SDK

Kotlin/Android bindings for connecting to mesh-llm meshes.

## GitHub Packages

Release workflow publishes the Android AAR to this repository's GitHub Packages Maven registry as:

```text
ai.meshllm:meshllm-android:<version>
```

Add the GitHub Packages Maven repository:

```kotlin
repositories {
    maven {
        url = uri("https://maven.pkg.github.com/Mesh-LLM/mesh-llm")
        credentials {
            username = providers.gradleProperty("gpr.user").orElse(System.getenv("GITHUB_ACTOR")).get()
            password = providers.gradleProperty("gpr.key").orElse(System.getenv("GITHUB_TOKEN")).get()
        }
    }
}
```

Then depend on the SDK:

```kotlin
dependencies {
    implementation("ai.meshllm:meshllm-android:0.1.0")
}
```

## Local Development

To build the Android artifact locally:

```bash
./gradlew assembleAar
```

This writes the AAR to `sdk/kotlin/build/outputs/aar/meshllm-android.aar`.

## Usage

```kotlin
val publicMeshes = MeshClient.discoverPublicMeshes()
val client = MeshClient.connectPublic(ownerKeypairBytesHex = loadPersistedOwnerKeypair())
client.join()
```

To join a specific public mesh, use the discovered invite token:

```kotlin
val publicMeshes = MeshClient.discoverPublicMeshes()
val specific = publicMeshes.first()
val client = MeshClient.connect(
    ownerKeypairBytesHex = loadPersistedOwnerKeypair(),
    inviteToken = specific.inviteToken,
)
client.join()
```
