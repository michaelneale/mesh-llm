import Foundation
import MeshLLM

enum ExampleError: Error {
    case noModels
    case noTokenDelta
    case didNotComplete
    case chatFailed(String)
}

protocol MeshExampleClient: Sendable {
    func join() async throws
    func listModels() async throws -> [Model]
    func chatStream(_ request: ChatRequest) -> AsyncThrowingStream<MeshEvent, Error>
    func disconnect() async
}

extension MeshClient: MeshExampleClient {}

struct MeshExampleRunner {
    let environment: [String: String]
    let now: @Sendable () -> Date
    let sleep: @Sendable (Duration) async throws -> Void
    let writeStdout: @Sendable (String) -> Void

    init(
        environment: [String: String] = ProcessInfo.processInfo.environment,
        now: @escaping @Sendable () -> Date = { Date() },
        sleep: @escaping @Sendable (Duration) async throws -> Void = { duration in
            try await Task.sleep(for: duration)
        },
        writeStdout: @escaping @Sendable (String) -> Void = { line in
            print(line)
        }
    ) {
        self.environment = environment
        self.now = now
        self.sleep = sleep
        self.writeStdout = writeStdout
    }

    func run(client: any MeshExampleClient) async throws {
        try await client.join()
        writeStdout("[connected]")

        let models = try await waitForModels(client)
        writeStdout("[models] N=\(models.count)")
        guard !models.isEmpty else {
            throw ExampleError.noModels
        }

        let requestedModel = environment["MESH_SDK_MODEL_ID"]
        let selectedModel = models.first(where: { $0.id == requestedModel })?.id ?? models[0].id
        let request = ChatRequest(
            model: selectedModel,
            messages: [ChatMessage(role: "user", content: "hello")]
        )

        let startTime = now()
        var firstToken = true
        var sawToken = false
        var completed = false
        chatLoop: for try await event in client.chatStream(request) {
            switch event {
            case .tokenDelta(_, let delta):
                if firstToken {
                    let ms = Int(now().timeIntervalSince(startTime) * 1000)
                    writeStdout("[chat] first_token_ms=\(ms)")
                    firstToken = false
                }
                sawToken = true
                writeStdout(delta)
            case .completed:
                completed = true
                writeStdout("[chat] done")
                break chatLoop
            case .failed(_, let error):
                throw ExampleError.chatFailed(error)
            default:
                break
            }
        }

        guard sawToken else {
            throw ExampleError.noTokenDelta
        }
        guard completed else {
            throw ExampleError.didNotComplete
        }

        await client.disconnect()
        writeStdout("[disconnect] ok")
    }

    func waitForModels(_ client: any MeshExampleClient) async throws -> [Model] {
        let deadline = now().addingTimeInterval(30)
        var lastError: Error?
        while now() < deadline {
            do {
                let models = try await client.listModels()
                if !models.isEmpty {
                    return models
                }
            } catch {
                lastError = error
            }
            try await sleep(.milliseconds(250))
        }
        if let lastError {
            throw lastError
        }
        return []
    }
}
