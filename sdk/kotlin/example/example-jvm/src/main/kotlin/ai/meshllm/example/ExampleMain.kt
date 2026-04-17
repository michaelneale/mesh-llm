package ai.meshllm.example

import ai.meshllm.ChatMessage
import ai.meshllm.ChatRequest
import ai.meshllm.Event
import ai.meshllm.MeshClient
import com.sun.jna.NativeLibrary
import kotlinx.coroutines.runBlocking
import uniffi.mesh_ffi.createClient
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

fun main(args: Array<String>) = runBlocking {
    val inviteToken = args.firstOrNull { !it.startsWith("--") }
    if (inviteToken == null) {
        System.err.println("Usage: ExampleMain <invite_token>")
        System.err.println("Set jna.library.path to the directory containing libmesh_ffi.")
        System.exit(1)
    }

    NativeLibrary.getInstance("mesh_ffi")
    // Generate an ephemeral owner keypair for the example. In a real app this
    // must be persisted across launches — see mesh-api-ffi::create_client docs.
    val ownerKeypairHex = uniffi.mesh_ffi.generateOwnerKeypairHex()
    val handle = createClient(ownerKeypairHex, inviteToken!!)
    val client = MeshClient(handle)

    client.join()
    println("[connected]")

    val models = waitForModels(client)
    println("[models] N=${models.size}")
    check(models.isNotEmpty()) { "mesh reported no models" }

    val selectedModel = System.getenv("MESH_SDK_MODEL_ID") ?: models.first().id

    val chatRequest = ChatRequest(
        model = selectedModel,
        messages = listOf(ChatMessage(role = "user", content = "hello")),
    )

    val latch = CountDownLatch(1)
    var firstTokenEmitted = false
    var completed = false
    var failed: String? = null
    val chatStartMs = System.currentTimeMillis()

    client.chat(chatRequest) { event ->
        when (event) {
            is Event.TokenDelta -> {
                if (!firstTokenEmitted) {
                    firstTokenEmitted = true
                    val elapsedMs = System.currentTimeMillis() - chatStartMs
                    println("[chat] first_token_ms=$elapsedMs")
                }
            }
            is Event.Completed -> {
                completed = true
                latch.countDown()
            }
            is Event.Failed -> {
                failed = event.error
                latch.countDown()
            }
            else -> Unit
        }
    }

    check(latch.await(60, TimeUnit.SECONDS)) { "chat timed out waiting for completion" }
    check(failed == null) { "chat failed: $failed" }
    check(firstTokenEmitted) { "chat emitted no token deltas" }
    check(completed) { "chat never completed" }
    println("[chat] done")

    client.disconnect()
    println("[disconnect] ok")
}

private fun waitForModels(client: MeshClient): List<ai.meshllm.Model> {
    val deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30)
    while (System.nanoTime() < deadline) {
        val models = runBlocking { client.listModels() }
        if (models.isNotEmpty()) {
            return models
        }
        Thread.sleep(250)
    }
    return emptyList()
}
