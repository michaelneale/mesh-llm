# Windows helpers

These optional PowerShell helpers wrap a local Windows build of `mesh-llm`.

Build first from the repository root:

```powershell
just build backend=vulkan
```

Start a local server:

```powershell
.\contrib\windows\StartMeshServer.ps1 -Model Qwen2.5-3B-Instruct-Q4_K_M -Device Vulkan1
```

Chat with that server:

```powershell
.\contrib\windows\StartChat.ps1 -Model Qwen2.5-3B-Instruct-Q4_K_M
```

The scripts default to `target\release\mesh-llm.exe` when it exists, otherwise
they fall back to `mesh-llm` on `PATH`.
