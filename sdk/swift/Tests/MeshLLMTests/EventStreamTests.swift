import XCTest
@testable import MeshLLM

final class EventStreamTests: XCTestCase {
    func testChatStreamEmitsTerminalEvent() async throws {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: makeOwnerKeypairBytesHex())
        let request = ChatRequest(model: "test", messages: [])

        var events: [MeshEvent] = []
        for try await event in client.chatStream(request) {
            events.append(event)
        }

        XCTAssertFalse(events.isEmpty)
        XCTAssertTrue(
            events.contains(where: isTerminalEvent),
            "Stream should emit a terminal event before finishing"
        )
    }

    func testResponsesStreamEmitsTerminalEvent() async throws {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: makeOwnerKeypairBytesHex())
        let request = ResponsesRequest(model: "test", input: "hello")

        var events: [MeshEvent] = []
        for try await event in client.responsesStream(request) {
            events.append(event)
        }

        XCTAssertFalse(events.isEmpty)
        XCTAssertTrue(
            events.contains(where: isTerminalEvent),
            "Stream should emit a terminal event before finishing"
        )
    }

    func testCancelOnTermination() async throws {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: makeOwnerKeypairBytesHex())
        let request = ChatRequest(model: "test", messages: [])

        for try await _ in client.chatStream(request) {
            break
        }
    }

    private func isTerminalEvent(_ event: MeshEvent) -> Bool {
        switch event {
        case .completed, .failed, .disconnected:
            return true
        default:
            return false
        }
    }
}
