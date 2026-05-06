import Foundation
import MeshLLM

let args = Array(CommandLine.arguments.dropFirst())
let inviteTokenArg = args.first { !$0.hasPrefix("--") }
guard let token = inviteTokenArg else {
    fputs("Usage: MeshExampleApp <invite_token>\n", stderr)
    exit(1)
}

// Generate an ephemeral owner keypair for the example. In a real app this
// must be persisted across launches — see mesh-api-ffi::create_client docs.
let ownerKeypairHex = generateOwnerKeypairBytesHex()
let client = MeshClient(inviteToken: InviteToken(token), ownerKeypairBytesHex: ownerKeypairHex)
let runner = MeshExampleRunner()

Task {
    do {
        try await runner.run(client: client)
    } catch {
        FileHandle.standardError.write(Data("[error] \(error)\n".utf8))
        exit(1)
    }
    exit(0)
}

RunLoop.main.run()
