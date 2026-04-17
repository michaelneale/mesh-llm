import XCTest
@testable import MeshLLM

final class EventStreamTests: XCTestCase {
    func testChatStreamEmitsCompletedEvent() async throws {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: generateOwnerKeypairHex())
        let request = ChatRequest(model: "test", messages: [])

        var events: [MeshEvent] = []
        for try await event in client.chatStream(request) {
            events.append(event)
        }

        XCTAssertFalse(events.isEmpty)
        let hasCompleted = events.contains { if case .completed = $0 { return true }; return false }
        XCTAssertTrue(hasCompleted, "Stream should emit Completed event")
    }

    func testResponsesStreamEmitsCompletedEvent() async throws {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: generateOwnerKeypairHex())
        let request = ResponsesRequest(model: "test", input: "hello")

        var events: [MeshEvent] = []
        for try await event in client.responsesStream(request) {
            events.append(event)
        }

        XCTAssertFalse(events.isEmpty)
        let hasCompleted = events.contains { if case .completed = $0 { return true }; return false }
        XCTAssertTrue(hasCompleted, "Stream should emit Completed event")
    }

    func testCancelOnTermination() async throws {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: generateOwnerKeypairHex())
        let request = ChatRequest(model: "test", messages: [])

        for try await _ in client.chatStream(request) {
            break
        }
    }
}
