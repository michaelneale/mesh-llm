package ai.meshllm

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.withContext
import uniffi.mesh_ffi.ChatMessageDto
import uniffi.mesh_ffi.ChatRequestDto
import uniffi.mesh_ffi.EventDto
import uniffi.mesh_ffi.EventListener as FfiEventListener
import uniffi.mesh_ffi.MeshClientHandleInterface
import uniffi.mesh_ffi.ModelDto
import uniffi.mesh_ffi.PublicMeshDto
import uniffi.mesh_ffi.PublicMeshQueryDto
import uniffi.mesh_ffi.ResponsesRequestDto
import uniffi.mesh_ffi.StatusDto

data class Model(val id: String, val name: String)

data class ChatMessage(val role: String, val content: String)

data class ChatRequest(val model: String, val messages: List<ChatMessage>)

data class ResponsesRequest(val model: String, val input: String)

data class Status(val connected: Boolean, val peerCount: ULong)

data class PublicMeshQuery(
    val model: String? = null,
    val minVramGb: Double? = null,
    val region: String? = null,
    val targetName: String? = null,
    val relays: List<String> = emptyList(),
)

data class PublicMesh(
    val inviteToken: String,
    val serving: List<String>,
    val wanted: List<String>,
    val onDisk: List<String>,
    val totalVramBytes: ULong,
    val nodeCount: ULong,
    val clientCount: ULong,
    val maxClients: ULong,
    val name: String?,
    val region: String?,
    val meshId: String?,
    val publisherNpub: String,
    val publishedAt: ULong,
    val expiresAt: ULong?,
)

@JvmInline
value class RequestId(val value: String)

sealed class Event {
    object Connecting : Event()
    data class Joined(val nodeId: String) : Event()
    data class ModelsUpdated(val models: List<Model>) : Event()
    data class TokenDelta(val requestId: RequestId, val delta: String) : Event()
    data class Completed(val requestId: RequestId) : Event()
    data class Failed(val requestId: RequestId, val error: String) : Event()
    data class Disconnected(val reason: String) : Event()
}

fun interface EventListener {
    fun onEvent(event: Event)
}

private fun ModelDto.toModel() = Model(id = id, name = name)

private fun StatusDto.toStatus() = Status(connected = connected, peerCount = peerCount)

private fun ChatMessage.toDto() = ChatMessageDto(role = role, content = content)

private fun ChatRequest.toDto() =
    ChatRequestDto(model = model, messages = messages.map { it.toDto() })

private fun ResponsesRequest.toDto() = ResponsesRequestDto(model = model, input = input)

private fun PublicMeshQuery.toDto() =
    PublicMeshQueryDto(
        model = model,
        minVramGb = minVramGb,
        region = region,
        targetName = targetName,
        relays = relays,
    )

private fun PublicMeshDto.toPublicMesh() =
    PublicMesh(
        inviteToken = inviteToken,
        serving = serving,
        wanted = wanted,
        onDisk = onDisk,
        totalVramBytes = totalVramBytes,
        nodeCount = nodeCount,
        clientCount = clientCount,
        maxClients = maxClients,
        name = name,
        region = region,
        meshId = meshId,
        publisherNpub = publisherNpub,
        publishedAt = publishedAt,
        expiresAt = expiresAt,
    )

private fun EventDto.toEvent(): Event =
    when (this) {
        is EventDto.Connecting -> Event.Connecting
        is EventDto.Joined -> Event.Joined(nodeId = nodeId)
        is EventDto.ModelsUpdated -> Event.ModelsUpdated(models = models.map { it.toModel() })
        is EventDto.TokenDelta -> Event.TokenDelta(requestId = RequestId(requestId), delta = delta)
        is EventDto.Completed -> Event.Completed(requestId = RequestId(requestId))
        is EventDto.Failed -> Event.Failed(requestId = RequestId(requestId), error = error)
        is EventDto.Disconnected -> Event.Disconnected(reason = reason)
    }

class MeshClient(private val handle: MeshClientHandleInterface) {
    companion object {
        suspend fun connect(
            ownerKeypairBytesHex: String,
            inviteToken: String,
        ): MeshClient =
            withContext(Dispatchers.IO) {
                MeshClient(uniffi.mesh_ffi.createClient(ownerKeypairBytesHex, inviteToken))
            }

        suspend fun discoverPublicMeshes(
            query: PublicMeshQuery = PublicMeshQuery(),
        ): List<PublicMesh> =
            withContext(Dispatchers.IO) {
                uniffi.mesh_ffi.discoverPublicMeshes(query.toDto()).map { it.toPublicMesh() }
            }

        suspend fun connectPublic(
            ownerKeypairBytesHex: String,
            query: PublicMeshQuery = PublicMeshQuery(),
        ): MeshClient =
            withContext(Dispatchers.IO) {
                MeshClient(
                    uniffi.mesh_ffi.createAutoClient(
                        ownerKeypairBytesHex,
                        query.toDto(),
                    ),
                )
            }
    }

    suspend fun join(): Unit = withContext(Dispatchers.IO) { handle.join() }

    suspend fun listModels(): List<Model> =
        withContext(Dispatchers.IO) { handle.listModels().map { it.toModel() } }

    fun chat(request: ChatRequest, listener: EventListener): RequestId {
        val bridge = object : FfiEventListener {
            override fun onEvent(event: EventDto) = listener.onEvent(event.toEvent())
        }
        return RequestId(handle.chat(request.toDto(), bridge))
    }

    fun responses(request: ResponsesRequest, listener: EventListener): RequestId {
        val bridge = object : FfiEventListener {
            override fun onEvent(event: EventDto) = listener.onEvent(event.toEvent())
        }
        return RequestId(handle.responses(request.toDto(), bridge))
    }

    fun cancel(requestId: RequestId) = handle.cancel(requestId.value)

    suspend fun status(): Status = withContext(Dispatchers.IO) { handle.status().toStatus() }

    suspend fun disconnect(): Unit = withContext(Dispatchers.IO) { handle.disconnect() }

    suspend fun reconnect(): Unit = withContext(Dispatchers.IO) { handle.reconnect() }

    fun chatFlow(request: ChatRequest): Flow<Event> = callbackFlow {
        var requestId: RequestId? = null
        var terminalRequestId: RequestId? = null
        requestId = chat(request) { event ->
            trySend(event)
            val currentRequestId = requestId
            if (currentRequestId != null && event.isTerminalFor(currentRequestId)) {
                terminalRequestId = currentRequestId
                close()
            } else if (currentRequestId == null) {
                terminalRequestId = event.terminalRequestId()
            }
        }
        if (requestId == terminalRequestId) {
            close()
        }
        awaitClose {
            val currentRequestId = requestId ?: return@awaitClose
            if (terminalRequestId != currentRequestId) {
                cancel(currentRequestId)
            }
        }
    }

    fun responsesFlow(request: ResponsesRequest): Flow<Event> = callbackFlow {
        var requestId: RequestId? = null
        var terminalRequestId: RequestId? = null
        requestId = responses(request) { event ->
            trySend(event)
            val currentRequestId = requestId
            if (currentRequestId != null && event.isTerminalFor(currentRequestId)) {
                terminalRequestId = currentRequestId
                close()
            } else if (currentRequestId == null) {
                terminalRequestId = event.terminalRequestId()
            }
        }
        if (requestId == terminalRequestId) {
            close()
        }
        awaitClose {
            val currentRequestId = requestId ?: return@awaitClose
            if (terminalRequestId != currentRequestId) {
                cancel(currentRequestId)
            }
        }
    }

    fun eventsFlow(): Flow<Event> = callbackFlow {
        val bridge = object : FfiEventListener {
            override fun onEvent(event: EventDto) {
                trySend(event.toEvent())
            }
        }
        val listenerId = handle.addEventListener(bridge)
        awaitClose {
            handle.removeEventListener(listenerId)
        }
    }
}

private fun Event.isTerminalFor(requestId: RequestId): Boolean =
    when (this) {
        is Event.Completed -> this.requestId == requestId
        is Event.Failed -> this.requestId == requestId
        else -> false
    }

private fun Event.terminalRequestId(): RequestId? =
    when (this) {
        is Event.Completed -> requestId
        is Event.Failed -> requestId
        else -> null
    }
