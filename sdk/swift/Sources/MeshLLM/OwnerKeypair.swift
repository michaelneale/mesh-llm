import Foundation

#if canImport(MeshLLMFFI)
import MeshLLMFFI

public func generateOwnerKeypairBytesHex() -> String {
    generateOwnerKeypairHex()
}
#else
public func generateOwnerKeypairBytesHex() -> String {
    ""
}
#endif
