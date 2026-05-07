// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "MeshExampleApp",
    platforms: [
        .macOS(.v13),
    ],
    dependencies: [
        .package(name: "mesh-llm", path: "../../../.."),
    ],
    targets: [
        .executableTarget(
            name: "MeshExampleApp",
            dependencies: [
                .product(name: "MeshLLM", package: "mesh-llm"),
            ],
            path: "Sources/MeshExampleApp"
        ),
    ]
)
