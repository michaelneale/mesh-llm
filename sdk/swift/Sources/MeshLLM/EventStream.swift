import Foundation

public extension MeshClient {
    func chatStream(_ request: ChatRequest) -> AsyncThrowingStream<MeshEvent, Error> {
        chat(request)
    }

    func responsesStream(_ request: ResponsesRequest) -> AsyncThrowingStream<MeshEvent, Error> {
        responses(request)
    }

    func observeEvents() -> AsyncStream<MeshEvent> {
        AsyncStream { continuation in
            do {
                let handle = try ensureHandle()
                let bridge = MeshEventObserverBridge(continuation: continuation)
                let listenerID = handle.addEventListener(listener: bridge)
                bridge.activate(listenerID: listenerID, handle: handle)
            } catch {
                continuation.finish()
            }
        }
    }
}

#if canImport(MeshLLMFFI)
import MeshLLMFFI

public final class EventStreamBridge: EventListener, @unchecked Sendable {
    private let continuation: AsyncThrowingStream<MeshEvent, Error>.Continuation
    private let onCancel: @Sendable (String) -> Void
    private let stateLock = NSLock()
    private var requestId: String?
    private var finished = false

    public init(
        continuation: AsyncThrowingStream<MeshEvent, Error>.Continuation,
        onCancel: @escaping @Sendable (String) -> Void
    ) {
        self.continuation = continuation
        self.onCancel = onCancel
        continuation.onTermination = { [weak self] _ in
            self?.cancelIfNeeded()
        }
    }

    public func activate(requestId: String) {
        stateLock.lock()
        guard !finished else {
            stateLock.unlock()
            return
        }
        self.requestId = requestId
        stateLock.unlock()
    }

    public func onEvent(event: EventDto) {
        let mapped = MeshClient.mapEvent(event)
        switch mapped {
        case .completed, .failed, .disconnected:
            finish(with: mapped)
        default:
            stateLock.lock()
            let isFinished = finished
            stateLock.unlock()
            guard !isFinished else {
                return
            }
            continuation.yield(mapped)
            break
        }
    }

    public func finish(throwing error: Error? = nil) {
        stateLock.lock()
        guard !finished else {
            stateLock.unlock()
            return
        }
        finished = true
        requestId = nil
        stateLock.unlock()

        if let error {
            continuation.finish(throwing: error)
        } else {
            continuation.finish()
        }
    }

    private func cancelIfNeeded() {
        stateLock.lock()
        guard !finished else {
            stateLock.unlock()
            return
        }
        let requestId = self.requestId
        finished = true
        self.requestId = nil
        stateLock.unlock()

        guard let requestId else {
            return
        }
        onCancel(requestId)
    }

    private func finish(with event: MeshEvent) {
        stateLock.lock()
        guard !finished else {
            stateLock.unlock()
            return
        }
        finished = true
        requestId = nil
        stateLock.unlock()

        continuation.yield(event)
        continuation.finish()
    }
}

public final class MeshEventObserverBridge: EventListener, @unchecked Sendable {
    private let continuation: AsyncStream<MeshEvent>.Continuation
    private let stateLock = NSLock()
    private weak var handle: MeshClientHandle?
    private var listenerID: String?
    private var finished = false

    public init(continuation: AsyncStream<MeshEvent>.Continuation) {
        self.continuation = continuation
        continuation.onTermination = { [weak self] _ in
            self?.removeIfNeeded()
        }
    }

    public func activate(listenerID: String, handle: MeshClientHandle) {
        stateLock.lock()
        guard !finished else {
            stateLock.unlock()
            return
        }
        self.listenerID = listenerID
        self.handle = handle
        stateLock.unlock()
    }

    public func onEvent(event: EventDto) {
        stateLock.lock()
        let isFinished = finished
        stateLock.unlock()
        guard !isFinished else {
            return
        }
        continuation.yield(MeshClient.mapEvent(event))
    }

    private func removeIfNeeded() {
        stateLock.lock()
        guard !finished else {
            stateLock.unlock()
            return
        }
        finished = true
        let listenerID = self.listenerID
        let handle = self.handle
        self.listenerID = nil
        self.handle = nil
        stateLock.unlock()

        guard let listenerID, let handle else {
            return
        }
        handle.removeEventListener(listenerId: listenerID)
    }
}
#else
public final class EventStreamBridge: @unchecked Sendable {
    private let continuation: AsyncThrowingStream<MeshEvent, Error>.Continuation
    private let requestId: RequestId
    private weak var client: MeshClient?
    private let stateLock = NSLock()
    private var finished = false

    public init(
        continuation: AsyncThrowingStream<MeshEvent, Error>.Continuation,
        requestId: RequestId,
        client: MeshClient
    ) {
        self.continuation = continuation
        self.requestId = requestId
        self.client = client

        continuation.onTermination = { [weak self] _ in
            guard let self else { return }
            self.client?.cancel(self.requestId)
        }
    }

    public func emit(_ event: MeshEvent) {
        stateLock.lock()
        guard !finished else {
            stateLock.unlock()
            return
        }

        switch event {
        case .completed, .failed, .disconnected:
            finished = true
            stateLock.unlock()
            continuation.yield(event)
            continuation.finish()
        default:
            stateLock.unlock()
            continuation.yield(event)
        }
    }

    public func finish(throwing error: Error? = nil) {
        stateLock.lock()
        guard !finished else {
            stateLock.unlock()
            return
        }
        finished = true
        stateLock.unlock()

        if let error {
            continuation.finish(throwing: error)
        } else {
            continuation.finish()
        }
    }
}
#endif
