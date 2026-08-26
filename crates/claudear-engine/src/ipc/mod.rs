//! Inter-process communication for the watcher daemon.
//!
//! Platform-specific transport details (Unix domain sockets vs TCP) live in
//! the [`transport`] module.  This file exposes the high-level helpers that the
//! rest of the crate uses for daemon lifecycle management.

mod client;
mod protocol;
mod server;
pub(crate) mod transport;

pub use client::{print_response, IpcClient};
pub use protocol::{IpcCommand, IpcData, IpcResponse, WatcherState};
pub use server::IpcServer;

use claudear_core::platform;
use std::path::PathBuf;

/// Returns a private runtime directory for IPC files, scoped to the current user.
///
/// * **Linux** — prefers `XDG_RUNTIME_DIR` (already user-private, mode 0700).
/// * **Windows** — uses `%LOCALAPPDATA%\claudear` or falls back to `%TEMP%\claudear`.
/// * **macOS / other** — creates `claudear-{uid}` under the system temp dir with
///   mode 0700.
fn ipc_runtime_dir() -> PathBuf {
    #[cfg(windows)]
    {
        // Prefer %LOCALAPPDATA%\claudear (user-private on Windows)
        if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
            let dir = PathBuf::from(local_app_data).join("claudear");
            if !dir.exists() {
                if let Err(e) = std::fs::create_dir_all(&dir) {
                    tracing::error!(
                        "Failed to create IPC runtime dir {:?}: {} — falling back to temp dir",
                        dir,
                        e
                    );
                    return std::env::temp_dir().join("claudear");
                }
            }
            return dir;
        }
        let dir = std::env::temp_dir().join("claudear");
        if !dir.exists() {
            let _ = std::fs::create_dir_all(&dir);
        }
        dir
    }

    #[cfg(not(windows))]
    {
        // On Linux, XDG_RUNTIME_DIR is already user-private
        if !cfg!(target_os = "macos") {
            if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
                return PathBuf::from(xdg);
            }
        }

        // Fallback: create a user-scoped subdirectory in the temp dir
        let uid = unsafe { libc::getuid() };
        let dir = std::env::temp_dir().join(format!("claudear-{}", uid));
        if !dir.exists() {
            if let Err(e) = std::fs::create_dir_all(&dir) {
                tracing::error!(
                    "Failed to create IPC runtime dir {:?}: {} — falling back to world-readable temp dir",
                    dir,
                    e
                );
                return std::env::temp_dir();
            }
            if let Err(e) = platform::set_dir_permissions_secure(&dir) {
                tracing::warn!("Failed to set IPC runtime dir permissions: {}", e);
            }
        }
        dir
    }
}

/// Default socket path for the IPC server (Unix) or port file path (Windows).
pub fn default_socket_path() -> PathBuf {
    #[cfg(windows)]
    {
        ipc_runtime_dir().join("claudear.port")
    }
    #[cfg(not(windows))]
    {
        ipc_runtime_dir().join("claudear.sock")
    }
}

/// Default PID file path.
pub fn default_pid_path() -> PathBuf {
    ipc_runtime_dir().join("claudear.pid")
}

/// Check if a watcher daemon is running.
pub fn is_daemon_running() -> bool {
    transport::check_connection(&default_socket_path())
}

/// Get the PID of the running daemon, if any.
pub fn get_daemon_pid() -> Option<u32> {
    let pid_path = default_pid_path();
    if pid_path.exists() {
        std::fs::read_to_string(&pid_path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
    } else {
        None
    }
}

/// Write the current process PID to the pid file.
pub fn write_pid_file() -> std::io::Result<()> {
    let pid_path = default_pid_path();
    std::fs::write(&pid_path, std::process::id().to_string())
}

/// Remove the PID file.
pub fn remove_pid_file() {
    let pid_path = default_pid_path();
    let _ = std::fs::remove_file(&pid_path);
}

/// Remove the socket file (Unix) or port file (Windows).
pub fn remove_socket_file() {
    let socket_path = default_socket_path();
    let _ = std::fs::remove_file(&socket_path);
}

/// Clean up stale socket/pid files from a previous crash.
pub fn cleanup_stale_files() {
    if let Some(pid) = get_daemon_pid() {
        if !platform::is_process_running(pid) {
            tracing::info!("Cleaning up stale files from previous run (PID {})", pid);
            remove_pid_file();
            remove_socket_file();
        }
    } else {
        // No PID file but socket/port file exists — check if stale
        let socket_path = default_socket_path();
        if transport::is_stale(&socket_path) {
            tracing::info!("Cleaning up stale socket file");
            remove_socket_file();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_socket_path_has_expected_extension() {
        let path = default_socket_path();
        #[cfg(windows)]
        assert!(
            path.ends_with("claudear.port"),
            "Expected socket path to end with 'claudear.port', got: {:?}",
            path
        );
        #[cfg(not(windows))]
        assert!(
            path.ends_with("claudear.sock"),
            "Expected socket path to end with 'claudear.sock', got: {:?}",
            path
        );
    }

    #[test]
    fn test_default_pid_path_ends_with_claudear_pid() {
        let path = default_pid_path();
        assert!(
            path.ends_with("claudear.pid"),
            "Expected pid path to end with 'claudear.pid', got: {:?}",
            path
        );
    }

    #[test]
    fn test_ipc_runtime_dir_is_absolute() {
        let dir = ipc_runtime_dir();
        assert!(
            dir.is_absolute(),
            "Expected ipc_runtime_dir to return an absolute path, got: {:?}",
            dir
        );
    }

    #[test]
    fn test_get_daemon_pid_returns_option() {
        let result = get_daemon_pid();
        if let Some(pid) = result {
            assert!(pid > 0, "PID should be positive, got: {}", pid);
        }
    }

    #[test]
    fn test_write_and_remove_pid_file_does_not_panic() {
        if is_daemon_running() {
            return;
        }

        let pid_path = default_pid_path();
        let existing_content = std::fs::read_to_string(&pid_path).ok();

        let write_result = write_pid_file();
        assert!(write_result.is_ok(), "write_pid_file should succeed");

        let read_pid = get_daemon_pid();
        assert_eq!(
            read_pid,
            Some(std::process::id()),
            "get_daemon_pid should return our process PID after write_pid_file"
        );

        remove_pid_file();

        if let Some(content) = existing_content {
            let _ = std::fs::write(&pid_path, content);
        } else {
            assert!(
                !pid_path.exists(),
                "PID file should be removed after remove_pid_file"
            );
        }
    }

    #[test]
    fn test_remove_socket_file_does_not_panic_when_no_socket() {
        if !is_daemon_running() {
            remove_socket_file();
        }
    }

    #[test]
    fn test_is_daemon_running_returns_bool() {
        let result = is_daemon_running();
        let _: bool = result;
    }

    #[test]
    fn test_cleanup_stale_files_does_not_panic() {
        cleanup_stale_files();
    }
}
