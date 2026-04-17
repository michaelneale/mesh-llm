import Foundation
import MeshLLM

enum ExampleError: Error {
    case noModels
    case noTokenDelta
    case didNotComplete
    case chatFailed(String)
}

let args = Array(CommandLine.arguments.dropFirst())
let inviteTokenArg = args.first { !$0.hasPrefix("--") }
guard let token = inviteTokenArg else {
    fputs("Usage: MeshExampleApp <invite_token>\n", stderr)
    exit(1)
}

// Generate an ephemeral owner keypair for the example. In a real app this
// must be persisted across launches — see mesh-api-ffi::create_client docs.
let ownerKeypairHex = generateOwnerKeypairHex()
let client = MeshClient(inviteToken: InviteToken(token), ownerKeypairBytesHex: ownerKeypairHex)

Task {
    do {
        try await client.join()
        print("[connected]")

        let models = try await waitForModels(client)
        print("[models] N=\(models.count)")
        guard !models.isEmpty else {
            throw ExampleError.noModels
        }

        let selectedModel = ProcessInfo.processInfo.environment["MESH_SDK_MODEL_ID"] ?? models[0].id
        let request = ChatRequest(
            model: selectedModel,
            messages: [ChatMessage(role: "user", content: "hello")]
        )

        let startTime = Date()
        var firstToken = true
        var sawToken = false
        var completed = false
        for try await event in client.chatStream(request) {
            switch event {
            case .tokenDelta(_, let delta):
                if firstToken {
                    let ms = Int(Date().timeIntervalSince(startTime) * 1000)
                    print("[chat] first_token_ms=\(ms)")
                    firstToken = false
                }
                sawToken = true
                print(delta, terminator: "")
            case .completed:
                completed = true
                print("\n[chat] done")
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
        print("[disconnect] ok")
    } catch {
        FileHandle.standardError.write(Data("[error] \(error)\n".utf8))
        exit(1)
    }
    exit(0)
}

RunLoop.main.run()

func waitForModels(_ client: MeshClient) async throws -> [Model] {
    let deadline = Date().addingTimeInterval(30)
    while Date() < deadline {
        let models = try await client.listModels()
        if !models.isEmpty {
            return models
        }
        try await Task.sleep(for: .milliseconds(250))
    }
    return []
}
