use anyhow::{bail, Context, Result};
#[cfg(unix)]
use std::ffi::CString;
use std::io;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

use crate::backend;
use crate::release_target::{CanonicalArch, CanonicalOs, ReleaseTarget};

const DEFAULT_RELEASE_REPO: &str = "Mesh-LLM/mesh-llm";
#[cfg(not(windows))]
const INSTALL_SCRIPT_URL: &str =
    "https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/main/install.sh";
const RELEASES_URL: &str = "https://github.com/Mesh-LLM/mesh-llm/releases/latest";
const SELF_UPDATE_ATTEMPTED_ENV: &str = "MESH_LLM_SELF_UPDATE_ATTEMPTED";
const SELF_UPDATE_REPO_ENV: &str = "MESH_LLM_SELF_UPDATE_REPO";

enum InstallOutcome {
    #[cfg_attr(windows, allow(dead_code))]
    RestartNow,
    #[cfg_attr(windows, allow(dead_code))]
    ExitNow,
    #[cfg_attr(not(windows), allow(dead_code))]
    HandoffAndExit,
}

#[derive(Clone, Copy)]
enum PostInstallAction {
    RestartCurrentProcess,
    ExitAfterInstall,
}

struct ReleaseInfo {
    tag: String,
    version: String,
    assets: Vec<String>,
}

struct UpdateTarget {
    exe: PathBuf,
    install_dir: PathBuf,
    release_target: ReleaseTarget,
    bundle_flavor: backend::BinaryFlavor,
}

#[derive(Clone, Copy, Debug)]
pub struct AutoUpdateOptions {
    pub auto_update: bool,
    pub plugin_requested: bool,
    pub command_is_update: bool,
    pub llama_flavor: Option<backend::BinaryFlavor>,
    pub current_version: &'static str,
}

#[derive(Clone, Copy, Debug)]
pub struct UpdateCommandOptions<'a> {
    pub llama_flavor: Option<backend::BinaryFlavor>,
    pub requested_version: Option<&'a str>,
    pub current_version: &'static str,
}

#[derive(Clone, Copy)]
enum ReleaseAssetPreference {
    StableFirst,
    VersionedFirst,
}

include!("entrypoints.rs");
include!("release_metadata.rs");
include!("update_target.rs");
include!("bundle_install.rs");
include!("tests.rs");
