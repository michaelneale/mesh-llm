import XCTest
@testable import MeshLLM

final class MeshClientTests: XCTestCase {
    func testClientCreation() {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: makeOwnerKeypairBytesHex())
        XCTAssertNotNil(client)
    }

    func testStatusBeforeJoin() async {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: makeOwnerKeypairBytesHex())
        let status = await client.status()
        XCTAssertFalse(status.connected)
    }

    func testJoinRejectsInvalidInviteToken() async {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: makeOwnerKeypairBytesHex())
        do {
            try await client.join()
            XCTFail("Joining with an invalid invite token should fail")
        } catch let error as FfiError {
            guard case .JoinFailed(let message) = error else {
                return XCTFail("Expected JoinFailed, got \(error)")
            }
            XCTAssertTrue(message.localizedCaseInsensitiveContains("invalid invite token"))
        } catch {
            XCTFail("Expected FfiError.JoinFailed, got \(error)")
        }

        let status = await client.status()
        XCTAssertFalse(status.connected)
    }

    func testDisconnectAfterFailedJoinLeavesClientDisconnected() async {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: makeOwnerKeypairBytesHex())
        do {
            try await client.join()
            XCTFail("Joining with an invalid invite token should fail")
        } catch {}

        await client.disconnect()
        let status = await client.status()
        XCTAssertFalse(status.connected)
    }

    func testReconnectRejectsInvalidInviteToken() async {
        let client = MeshClient(inviteToken: InviteToken("test-token"), ownerKeypairBytesHex: makeOwnerKeypairBytesHex())
        do {
            try await client.reconnect()
            XCTFail("Reconnect with an invalid invite token should fail")
        } catch let error as FfiError {
            guard case .ReconnectFailed(let message) = error else {
                return XCTFail("Expected ReconnectFailed, got \(error)")
            }
            XCTAssertTrue(message.localizedCaseInsensitiveContains("invalid invite token"))
        } catch {
            XCTFail("Expected FfiError.ReconnectFailed, got \(error)")
        }

        let status = await client.status()
        XCTAssertFalse(status.connected)
    }
}
