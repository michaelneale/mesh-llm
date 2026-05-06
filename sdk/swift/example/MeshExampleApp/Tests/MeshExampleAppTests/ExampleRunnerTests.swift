import XCTest
import MeshLLM
@testable import MeshExampleApp

final class ExampleRunnerTests: XCTestCase {
    func testRunnerCompletesExampleFlow() async throws {
        let client = MockMeshExampleClient(
            modelResponses: [
                [],
                [Model(id: "mesh-model-1", name: "mesh-model-1")]
            ],
            streamedEvents: [
                .tokenDelta(requestId: "req-1", delta: "hello from mesh"),
                .completed(requestId: "req-1")
            ]
        )
        let output = LockedLines()
        let runner = MeshExampleRunner(
            environment: [:],
            sleep: { _ in },
            writeStdout: { line in
                output.append(line)
            }
        )

        try await runner.run(client: client)

        let joinCallCount = await client.joinCallCount
        let disconnectCallCount = await client.disconnectCallCount
        let selectedModel = await client.lastChatRequest?.model
        XCTAssertEqual(joinCallCount, 1)
        XCTAssertEqual(disconnectCallCount, 1)
        XCTAssertEqual(selectedModel, "mesh-model-1")
        XCTAssertEqual(output.lines, [
            "[connected]",
            "[models] N=1",
            "[chat] first_token_ms=0",
            "hello from mesh",
            "[chat] done",
            "[disconnect] ok",
        ])
    }

    func testRunnerHonorsModelOverrideFromEnvironment() async throws {
        let client = MockMeshExampleClient(
            modelResponses: [[
                Model(id: "mesh-model-1", name: "mesh-model-1"),
                Model(id: "mesh-model-2", name: "mesh-model-2"),
            ]],
            streamedEvents: [
                .tokenDelta(requestId: "req-1", delta: "hi"),
                .completed(requestId: "req-1")
            ]
        )
        let runner = MeshExampleRunner(
            environment: ["MESH_SDK_MODEL_ID": "mesh-model-2"],
            sleep: { _ in },
            writeStdout: { _ in }
        )

        try await runner.run(client: client)

        let selectedModel = await client.lastChatRequest?.model
        XCTAssertEqual(selectedModel, "mesh-model-2")
    }

    func testRunnerRetriesTransientModelListingFailures() async throws {
        let client = MockMeshExampleClient(
            modelResponses: [
                .failure(TestError.transient),
                .success([]),
                .success([Model(id: "mesh-model-1", name: "mesh-model-1")]),
            ],
            streamedEvents: [
                .tokenDelta(requestId: "req-1", delta: "hello"),
                .completed(requestId: "req-1"),
            ]
        )
        let runner = MeshExampleRunner(
            environment: [:],
            sleep: { _ in },
            writeStdout: { _ in }
        )

        try await runner.run(client: client)

        let selectedModel = await client.lastChatRequest?.model
        XCTAssertEqual(selectedModel, "mesh-model-1")
    }
}

private actor MockMeshExampleClient: MeshExampleClient {
    private var remainingModelResponses: [Result<[Model], Error>]
    private let streamedEvents: [MeshEvent]
    private(set) var joinCallCount = 0
    private(set) var disconnectCallCount = 0
    private(set) var lastChatRequest: ChatRequest?

    init(modelResponses: [[Model]], streamedEvents: [MeshEvent]) {
        self.remainingModelResponses = modelResponses.map(Result.success)
        self.streamedEvents = streamedEvents
    }

    init(modelResponses: [Result<[Model], Error>], streamedEvents: [MeshEvent]) {
        self.remainingModelResponses = modelResponses
        self.streamedEvents = streamedEvents
    }

    func join() async throws {
        joinCallCount += 1
    }

    func listModels() async throws -> [Model] {
        if remainingModelResponses.isEmpty {
            return []
        }
        return try remainingModelResponses.removeFirst().get()
    }

    nonisolated func chatStream(_ request: ChatRequest) -> AsyncThrowingStream<MeshEvent, Error> {
        AsyncThrowingStream { continuation in
            Task {
                await self.setLastChatRequest(request)
                for event in await self.streamedEventsSnapshot() {
                    continuation.yield(event)
                }
                continuation.finish()
            }
        }
    }

    func disconnect() async {
        disconnectCallCount += 1
    }

    private func setLastChatRequest(_ request: ChatRequest) {
        lastChatRequest = request
    }

    private func streamedEventsSnapshot() -> [MeshEvent] {
        streamedEvents
    }
}

private enum TestError: Error {
    case transient
}

private final class LockedLines: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []

    func append(_ line: String) {
        lock.lock()
        storage.append(line)
        lock.unlock()
    }

    var lines: [String] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}
