//! Automatic port forwarding for privileged ports (< 1024).
//!
//! When running without root privileges, privileged ports cannot be bound
//! directly. This module detects the OS and delegates to the appropriate
//! platform backend:
//!
//! - **macOS**: [`macos`] — `pfctl` rules in a pf anchor.
//! - **Linux**: [`linux`] — `iptables` NAT REDIRECT rules.
//!
//! Rules are cleaned up automatically when the returned [`PortForwardGuard`]
//! is dropped.

#[cfg(target_os = "linux")]
pub mod linux;
#[cfg(target_os = "macos")]
pub mod macos;

use std::process::Command;

/// Offset added to privileged ports to get the actual bind port.
const PORT_OFFSET: u16 = 8000;

/// Guard that cleans up port forwarding rules when dropped.
///
/// Hold this value alive for the lifetime of the server. When dropped
/// (including on panic unwind), the platform-specific rules are removed.
pub struct PortForwardGuard {
    redirects: Vec<(u16, u16)>,
}

impl Drop for PortForwardGuard {
    fn drop(&mut self) {
        if self.redirects.is_empty() {
            return;
        }

        if let Err(e) = platform_cleanup(&self.redirects) {
            tracing::warn!(error = %e, "Failed to clean up port forwarding rules");
            for hint in platform_cleanup_hint(&self.redirects) {
                tracing::warn!("Run manually: {hint}");
            }
        }
    }
}

/// Returns `true` if the given port is privileged and we are running
/// on a supported OS as a non-root user (i.e. we cannot bind to it directly).
pub fn needs_port_forward(port: u16) -> bool {
    is_supported_platform() && port < 1024 && !is_root()
}

/// Get the high port to actually bind to when forwarding is needed.
pub fn forwarded_port(configured_port: u16) -> u16 {
    configured_port.saturating_add(PORT_OFFSET)
}

/// Set up port forwarding rules for the current platform.
///
/// Each entry in `redirects` is `(from_port, to_port)` — traffic arriving
/// on `from_port` is redirected to `to_port` where the app is actually
/// listening. The user will be prompted for their sudo password once.
///
/// Returns a [`PortForwardGuard`] that cleans up the rules when dropped.
pub fn setup_port_forward(
    redirects: &[(u16, u16)],
    bind_address: &str,
) -> Result<PortForwardGuard, String> {
    if redirects.is_empty() {
        return Ok(PortForwardGuard {
            redirects: Vec::new(),
        });
    }

    ensure_sudo()?;

    tracing::info!(
        "Setting up port forwarding:\n{}",
        redirects
            .iter()
            .map(|(f, t)| format!("  :{f} -> :{t}"))
            .collect::<Vec<_>>()
            .join("\n")
    );

    platform_setup(redirects, bind_address)?;

    Ok(PortForwardGuard {
        redirects: redirects.to_vec(),
    })
}

// ---------------------------------------------------------------------------
// Platform dispatch
// ---------------------------------------------------------------------------

fn is_supported_platform() -> bool {
    cfg!(target_os = "macos") || cfg!(target_os = "linux")
}

#[cfg(target_os = "macos")]
fn platform_setup(redirects: &[(u16, u16)], bind_address: &str) -> Result<(), String> {
    macos::setup(redirects, bind_address)
}

#[cfg(target_os = "linux")]
fn platform_setup(redirects: &[(u16, u16)], bind_address: &str) -> Result<(), String> {
    linux::setup(redirects, bind_address)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn platform_setup(_redirects: &[(u16, u16)], _bind_address: &str) -> Result<(), String> {
    Err("Port forwarding is only supported on macOS and Linux".to_string())
}

#[cfg(target_os = "macos")]
fn platform_cleanup(redirects: &[(u16, u16)]) -> Result<(), String> {
    macos::cleanup(redirects)
}

#[cfg(target_os = "linux")]
fn platform_cleanup(redirects: &[(u16, u16)]) -> Result<(), String> {
    linux::cleanup(redirects)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn platform_cleanup(_redirects: &[(u16, u16)]) -> Result<(), String> {
    Ok(())
}

#[cfg(target_os = "macos")]
fn platform_cleanup_hint(redirects: &[(u16, u16)]) -> Vec<String> {
    macos::cleanup_hint(redirects)
}

#[cfg(target_os = "linux")]
fn platform_cleanup_hint(redirects: &[(u16, u16)]) -> Vec<String> {
    linux::cleanup_hint(redirects)
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn platform_cleanup_hint(_redirects: &[(u16, u16)]) -> Vec<String> {
    Vec::new()
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Pre-authenticate sudo so subsequent calls reuse the cached credential.
fn ensure_sudo() -> Result<(), String> {
    tracing::info!("Port forwarding requires sudo — you may be prompted for your password");

    let status = Command::new("sudo")
        .arg("-v")
        .status()
        .map_err(|e| format!("Failed to run sudo: {e}"))?;

    if !status.success() {
        return Err("sudo authentication failed — cannot set up port forwarding".to_string());
    }

    Ok(())
}

/// Check if the current process is running as root.
fn is_root() -> bool {
    Command::new("id")
        .arg("-u")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim() == "0")
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // forwarded_port
    // -----------------------------------------------------------------------

    #[test]
    fn forwarded_port_standard_https() {
        assert_eq!(forwarded_port(443), 8443);
    }

    #[test]
    fn forwarded_port_standard_http() {
        assert_eq!(forwarded_port(80), 8080);
    }

    #[test]
    fn forwarded_port_zero() {
        assert_eq!(forwarded_port(0), PORT_OFFSET);
    }

    #[test]
    fn forwarded_port_one() {
        assert_eq!(forwarded_port(1), PORT_OFFSET + 1);
    }

    #[test]
    fn forwarded_port_boundary_1023() {
        assert_eq!(forwarded_port(1023), 1023 + PORT_OFFSET);
    }

    #[test]
    fn forwarded_port_boundary_1024() {
        assert_eq!(forwarded_port(1024), 1024 + PORT_OFFSET);
    }

    #[test]
    fn forwarded_port_high_port() {
        assert_eq!(forwarded_port(3100), 3100 + PORT_OFFSET);
    }

    #[test]
    fn forwarded_port_saturates_at_u16_max() {
        assert_eq!(forwarded_port(u16::MAX), u16::MAX);
        assert_eq!(forwarded_port(u16::MAX - 1), u16::MAX);
        assert_eq!(forwarded_port(u16::MAX - PORT_OFFSET + 1), u16::MAX);
    }

    #[test]
    fn forwarded_port_just_below_saturation() {
        let max_non_saturating = u16::MAX - PORT_OFFSET;
        assert_eq!(forwarded_port(max_non_saturating), u16::MAX);
    }

    // -----------------------------------------------------------------------
    // needs_port_forward
    // -----------------------------------------------------------------------

    #[test]
    fn needs_port_forward_high_ports_never_forwarded() {
        assert!(!needs_port_forward(1024));
        assert!(!needs_port_forward(3100));
        assert!(!needs_port_forward(8080));
        assert!(!needs_port_forward(8443));
        assert!(!needs_port_forward(u16::MAX));
    }

    #[test]
    fn needs_port_forward_privileged_ports_on_supported_os() {
        if is_supported_platform() && !is_root() {
            assert!(needs_port_forward(80));
            assert!(needs_port_forward(443));
            assert!(needs_port_forward(1));
            assert!(needs_port_forward(1023));
        }
    }

    #[test]
    fn needs_port_forward_port_zero() {
        if is_supported_platform() && !is_root() {
            assert!(needs_port_forward(0));
        }
    }

    // -----------------------------------------------------------------------
    // is_root
    // -----------------------------------------------------------------------

    #[test]
    fn is_root_consistent() {
        assert_eq!(is_root(), is_root());
    }

    // -----------------------------------------------------------------------
    // is_supported_platform
    // -----------------------------------------------------------------------

    #[test]
    fn is_supported_platform_matches_cfg() {
        let expected = cfg!(target_os = "macos") || cfg!(target_os = "linux");
        assert_eq!(is_supported_platform(), expected);
    }

    // -----------------------------------------------------------------------
    // Guard
    // -----------------------------------------------------------------------

    #[test]
    fn guard_empty_does_not_panic() {
        drop(PortForwardGuard {
            redirects: Vec::new(),
        });
    }

    #[test]
    fn guard_stores_redirects() {
        let guard = PortForwardGuard {
            redirects: vec![(443, 8443), (80, 8080)],
        };
        assert_eq!(guard.redirects, vec![(443, 8443), (80, 8080)]);
        // Drop may fail (no sudo) but must not panic.
        drop(guard);
    }

    #[test]
    fn guard_with_nonexistent_rules_does_not_panic() {
        // Dropping a guard whose rules were never actually installed
        // should log a warning but never panic.
        drop(PortForwardGuard {
            redirects: vec![(12345, 20345)],
        });
    }

    #[test]
    fn setup_empty_returns_empty_guard() {
        let guard = setup_port_forward(&[], "127.0.0.1").unwrap();
        assert!(guard.redirects.is_empty());
    }

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------

    #[test]
    fn port_offset_is_8000() {
        assert_eq!(PORT_OFFSET, 8000);
    }

    // -----------------------------------------------------------------------
    // Platform cleanup hint
    // -----------------------------------------------------------------------

    #[test]
    fn cleanup_hint_not_empty_for_nonempty_redirects() {
        if is_supported_platform() {
            let hints = platform_cleanup_hint(&[(443, 8443)]);
            assert!(!hints.is_empty());
            // Each hint should be a valid shell command.
            for hint in &hints {
                assert!(hint.starts_with("sudo "));
            }
        }
    }

    #[test]
    fn cleanup_hint_empty_for_empty_redirects() {
        let hints = platform_cleanup_hint(&[]);
        assert!(hints.is_empty());
    }
}
