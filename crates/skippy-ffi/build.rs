fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_STAGE_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_STAGE_LIB_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_STAGE_LINK_MODE");
    println!("cargo:rerun-if-env-changed=SKIPPY_LLAMA_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=SKIPPY_LLAMA_LIB_DIR");
    println!("cargo:rerun-if-env-changed=SKIPPY_LLAMA_LINK_MODE");

    let link_mode =
        std::env::var("LLAMA_STAGE_LINK_MODE").or_else(|_| std::env::var("SKIPPY_LLAMA_LINK_MODE"));
    if link_mode.as_deref() == Ok("dynamic") {
        if let Ok(lib_dir) =
            std::env::var("LLAMA_STAGE_LIB_DIR").or_else(|_| std::env::var("SKIPPY_LLAMA_LIB_DIR"))
        {
            println!("cargo:rustc-link-search=native={lib_dir}");
        }
        println!("cargo:rustc-link-lib=dylib=mtmd");
        println!("cargo:rustc-link-lib=dylib=llama-common");
        println!("cargo:rustc-link-lib=dylib=llama");
        return;
    }

    let workspace_root = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set"),
    )
    .join("../..");
    let build_dir = std::env::var("LLAMA_STAGE_BUILD_DIR")
        .or_else(|_| std::env::var("SKIPPY_LLAMA_BUILD_DIR"))
        .map(std::path::PathBuf::from)
        .map(|path| {
            if path.is_absolute() {
                path
            } else {
                workspace_root.join(path)
            }
        })
        .unwrap_or_else(|_| workspace_root.join(".deps/llama.cpp/build-stage-abi-static"));

    let search_dirs = [
        build_dir.join("tools/mtmd"),
        build_dir.join("common"),
        build_dir.join("src"),
        build_dir.join("ggml/src"),
        build_dir.join("ggml/src/ggml-cpu"),
        build_dir.join("ggml/src/ggml-blas"),
        build_dir.join("ggml/src/ggml-cuda"),
        build_dir.join("ggml/src/ggml-hip"),
        build_dir.join("ggml/src/ggml-metal"),
        build_dir.join("ggml/src/ggml-vulkan"),
    ];

    for dir in search_dirs.iter().filter(|dir| dir.exists()) {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    let cmake_cache = build_dir.join("CMakeCache.txt");
    if cmake_cache.exists() {
        println!("cargo:rerun-if-changed={}", cmake_cache.display());
    }

    for archive in [
        build_dir.join("src/libllama.a"),
        build_dir.join("tools/mtmd/libmtmd.a"),
        build_dir.join("common/libllama-common.a"),
        build_dir.join("common/libllama-common-base.a"),
        build_dir.join("ggml/src/libggml.a"),
        build_dir.join("ggml/src/libggml-base.a"),
        build_dir.join("ggml/src/ggml-cpu/libggml-cpu.a"),
        build_dir.join("ggml/src/ggml-blas/libggml-blas.a"),
        build_dir.join("ggml/src/ggml-cuda/libggml-cuda.a"),
        build_dir.join("ggml/src/ggml-hip/libggml-hip.a"),
        build_dir.join("ggml/src/ggml-metal/libggml-metal.a"),
        build_dir.join("ggml/src/ggml-vulkan/libggml-vulkan.a"),
    ]
    .iter()
    .filter(|archive| archive.exists())
    {
        println!("cargo:rerun-if-changed={}", archive.display());
    }

    if build_dir.join("tools/mtmd/libmtmd.a").exists() {
        println!("cargo:rustc-link-lib=static=mtmd");
    }
    println!("cargo:rustc-link-lib=static=llama-common");
    println!("cargo:rustc-link-lib=static=llama-common-base");
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml");
    let has_cuda = build_dir.join("ggml/src/ggml-cuda/libggml-cuda.a").exists();
    if has_cuda {
        println!("cargo:rustc-link-lib=static=ggml-cuda");
    }
    let has_hip = build_dir.join("ggml/src/ggml-hip/libggml-hip.a").exists();
    if has_hip {
        println!("cargo:rustc-link-lib=static=ggml-hip");
    }
    let has_vulkan = build_dir
        .join("ggml/src/ggml-vulkan/libggml-vulkan.a")
        .exists();
    if has_vulkan {
        println!("cargo:rustc-link-lib=static=ggml-vulkan");
    }
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    if build_dir.join("ggml/src/ggml-blas/libggml-blas.a").exists() {
        println!("cargo:rustc-link-lib=static=ggml-blas");
    }
    if build_dir
        .join("ggml/src/ggml-metal/libggml-metal.a")
        .exists()
    {
        println!("cargo:rustc-link-lib=static=ggml-metal");
    }
    println!("cargo:rustc-link-lib=static=ggml-base");

    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("apple-darwin") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        if build_dir
            .join("ggml/src/ggml-metal/libggml-metal.a")
            .exists()
        {
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
    } else if target.contains("linux") {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=dylib=m");
        println!("cargo:rustc-link-lib=dylib=dl");
        println!("cargo:rustc-link-lib=dylib=pthread");
        for lib in linux_openmp_libs(&cmake_cache) {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
        if has_cuda {
            link_linux_cuda_libs(&cmake_cache);
        }
        if has_hip {
            link_linux_hip_libs();
        }
        if has_vulkan {
            println!("cargo:rustc-link-lib=dylib=vulkan");
        }
    }
}

fn link_linux_cuda_libs(cmake_cache: &std::path::Path) {
    for (cache_key, lib) in [
        ("CUDA_cuda_driver_LIBRARY", "cuda"),
        ("CUDA_cudart_LIBRARY", "cudart"),
        ("CUDA_cublas_LIBRARY", "cublas"),
        ("CUDA_cublasLt_LIBRARY", "cublasLt"),
    ] {
        link_linux_lib_from_cache(cmake_cache, cache_key, lib);
    }
    // NCCL is conditionally linked by CMake when found on the system.
    // Check CMakeCache for NCCL_LIBRARY to detect this and extract the search path.
    if let Ok(contents) = std::fs::read_to_string(cmake_cache) {
        if let Some(nccl_path) = cmake_cache_value(&contents, "NCCL_LIBRARY") {
            if !nccl_path.contains("NOTFOUND") {
                let path = std::path::PathBuf::from(&nccl_path);
                if let Some(parent) = path.parent() {
                    if parent.is_dir() {
                        println!("cargo:rustc-link-search=native={}", parent.display());
                    }
                }
                println!("cargo:rustc-link-lib=dylib=nccl");
            }
        }
    }
}

fn link_linux_hip_libs() {
    // Add ROCm library search paths
    for search_path in ["/opt/rocm/lib", "/opt/rocm/hip/lib"] {
        if std::path::Path::new(search_path).is_dir() {
            println!("cargo:rustc-link-search=native={search_path}");
        }
    }
    for lib in ["amdhip64", "rocblas", "hipblas"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    // RCCL (ROCm Collective Communications Library) provides the NCCL interface.
    if std::path::Path::new("/opt/rocm/lib/librccl.so").exists() {
        println!("cargo:rustc-link-lib=dylib=rccl");
    }
}

fn link_linux_lib_from_cache(cmake_cache: &std::path::Path, cache_key: &str, lib: &str) {
    if let Ok(cache) = std::fs::read_to_string(cmake_cache) {
        if let Some(path) = cmake_cache_value(&cache, cache_key) {
            let path = std::path::PathBuf::from(path);
            if path.exists() {
                if let Some(parent) = path.parent() {
                    println!("cargo:rustc-link-search=native={}", parent.display());
                }
            }
        }
    }
    println!("cargo:rustc-link-lib=dylib={lib}");
}

fn linux_openmp_libs(cmake_cache: &std::path::Path) -> Vec<String> {
    let Ok(cache) = std::fs::read_to_string(cmake_cache) else {
        return Vec::new();
    };

    let mut libs = Vec::new();
    for key in ["OpenMP_C_LIB_NAMES", "OpenMP_CXX_LIB_NAMES"] {
        if let Some(value) = cmake_cache_value(&cache, key) {
            for lib in value.split(';') {
                let lib = lib.trim();
                if lib.is_empty() || lib == "NOTFOUND" || lib == "pthread" {
                    continue;
                }
                if !libs.iter().any(|existing| existing == lib) {
                    libs.push(lib.to_string());
                }
            }
        }
    }

    if libs.is_empty() && cmake_cache_bool(&cache, "GGML_OPENMP_ENABLED") {
        libs.push("gomp".to_string());
    }

    libs
}

fn cmake_cache_value(cache: &str, key: &str) -> Option<String> {
    cache.lines().find_map(|line| {
        let (lhs, rhs) = line.split_once('=')?;
        let (name, _) = lhs.split_once(':')?;
        (name == key).then(|| rhs.to_string())
    })
}

fn cmake_cache_bool(cache: &str, key: &str) -> bool {
    cmake_cache_value(cache, key)
        .map(|value| matches!(value.as_str(), "ON" | "TRUE" | "1"))
        .unwrap_or(false)
}
