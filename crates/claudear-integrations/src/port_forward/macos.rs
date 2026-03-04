//! macOS port forwarding via `pfctl` (packet filter).
//!
//! Uses the `com.apple/claudear` pf anchor, which is evaluated by the default
//! macOS `/etc/pf.conf` rule `rdr-anchor "com.apple/*"`.

use std::io::Write;
use std::process::{Command, Stdio};

/// pf anchor name. Nested under `com.apple` so it is evaluated by the
/// default macOS pf.conf `rdr-anchor "com.apple/*"` rule without needing
/// to modify `/etc/pf.conf`.
pub const PF_ANCHOR: &str = "com.apple/claudear";

/// Apply pf redirect rules for the given redirects and bind address.
pub fn setup(redirects: &[(u16, u16)], bind_address: &str) -> Result<(), String> {
    let interfaces = get_interfaces(bind_address);
    let rules = build_rules(redirects, &interfaces);

    // Load rules into our pf anchor.
    let mut child = Command::new("sudo")
        .args(["pfctl", "-a", PF_ANCHOR, "-f", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to run sudo pfctl: {e}"))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(rules.as_bytes())
            .map_err(|e| format!("Failed to write pf rules: {e}"))?;
    }

    let output = child
        .wait_with_output()
        .map_err(|e| format!("Failed to wait for pfctl: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("pfctl failed to load rules: {stderr}"));
    }

    // Enable pf if not already enabled (ignore errors — may already be active).
    let _ = Command::new("sudo")
        .args(["pfctl", "-e"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();

    tracing::info!("macOS pf port forwarding active (anchor: {PF_ANCHOR})");
    Ok(())
}

/// Flush our pf anchor. Uses `sudo -n` (non-interactive) so cleanup
/// never blocks on a password prompt during shutdown.
pub fn cleanup(_redirects: &[(u16, u16)]) -> Result<(), String> {
    tracing::info!("Cleaning up macOS pf port forwarding rules");

    let output = Command::new("sudo")
        .args(["-n", "pfctl", "-a", PF_ANCHOR, "-F", "all"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to run pfctl cleanup: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("pfctl cleanup failed: {stderr}"));
    }

    Ok(())
}

/// Return manual cleanup commands the user can run.
pub fn cleanup_hint(redirects: &[(u16, u16)]) -> Vec<String> {
    if redirects.is_empty() {
        return Vec::new();
    }
    vec![format!("sudo pfctl -a {PF_ANCHOR} -F all")]
}

// ---------------------------------------------------------------------------
// Pure helpers (testable without sudo)
// ---------------------------------------------------------------------------

/// Build the pf rules string for the given redirects and interfaces.
pub fn build_rules(redirects: &[(u16, u16)], interfaces: &[String]) -> String {
    let mut rules = String::new();
    for (from_port, to_port) in redirects {
        for iface in interfaces {
            rules.push_str(&format!(
                "rdr pass on {iface} inet proto tcp from any to any port {from_port} -> 127.0.0.1 port {to_port}\n"
            ));
        }
    }
    rules
}

/// Determine which pf interfaces to redirect on.
pub fn get_interfaces(bind_address: &str) -> Vec<String> {
    let mut interfaces = vec!["lo0".to_string()];

    // If binding to all interfaces, also redirect on the default network
    // interface so that external traffic on the privileged port is forwarded.
    if bind_address == "0.0.0.0" || bind_address == "::" {
        if let Some(iface) = get_default_interface() {
            if iface != "lo0" {
                interfaces.push(iface);
            }
        }
    }

    interfaces
}

/// Get the default network interface from the macOS routing table.
pub fn get_default_interface() -> Option<String> {
    let output = Command::new("route")
        .args(["-n", "get", "default"])
        .output()
        .ok()?;

    let stdout = String::from_utf8(output.stdout).ok()?;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if let Some(iface) = trimmed.strip_prefix("interface:") {
            return Some(iface.trim().to_string());
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Rule generation
    // -----------------------------------------------------------------------

    #[test]
    fn single_redirect_single_interface() {
        let rules = build_rules(&[(443, 8443)], &["lo0".into()]);
        assert_eq!(
            rules,
            "rdr pass on lo0 inet proto tcp from any to any port 443 -> 127.0.0.1 port 8443\n"
        );
    }

    #[test]
    fn single_redirect_multiple_interfaces() {
        let rules = build_rules(&[(443, 8443)], &["lo0".into(), "en0".into()]);
        let lines: Vec<&str> = rules.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("lo0"));
        assert!(lines[1].contains("en0"));
        for line in &lines {
            assert!(line.contains("port 443"));
            assert!(line.contains("port 8443"));
        }
    }

    #[test]
    fn multiple_redirects_single_interface() {
        let rules = build_rules(&[(443, 8443), (80, 8080)], &["lo0".into()]);
        let lines: Vec<&str> = rules.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("port 443") && lines[0].contains("port 8443"));
        assert!(lines[1].contains("port 80") && lines[1].contains("port 8080"));
    }

    #[test]
    fn multiple_redirects_multiple_interfaces() {
        let rules = build_rules(&[(443, 8443), (80, 8080)], &["lo0".into(), "en0".into()]);
        let lines: Vec<&str> = rules.lines().collect();
        assert_eq!(lines.len(), 4); // 2 redirects × 2 interfaces
    }

    #[test]
    fn empty_redirects() {
        assert!(build_rules(&[], &["lo0".into()]).is_empty());
    }

    #[test]
    fn empty_interfaces() {
        assert!(build_rules(&[(443, 8443)], &[]).is_empty());
    }

    #[test]
    fn valid_pf_syntax() {
        let rules = build_rules(&[(443, 8443)], &["lo0".into()]);
        let line = rules.lines().next().unwrap();
        assert!(line.starts_with("rdr pass on "));
        assert!(line.contains("inet proto tcp"));
        assert!(line.contains("from any to any"));
        assert!(line.contains("-> 127.0.0.1"));
    }

    #[test]
    fn port_zero() {
        let rules = build_rules(&[(0, 8000)], &["lo0".into()]);
        assert!(rules.contains("port 0 ->"));
        assert!(rules.contains("port 8000"));
    }

    #[test]
    fn max_port() {
        let rules = build_rules(&[(1023, 65535)], &["lo0".into()]);
        assert!(rules.contains("port 1023"));
        assert!(rules.contains("port 65535"));
    }

    // -----------------------------------------------------------------------
    // Interface selection (requires macOS routing table)
    // -----------------------------------------------------------------------

    #[test]
    fn interfaces_loopback_only() {
        let ifaces = get_interfaces("127.0.0.1");
        assert_eq!(ifaces, vec!["lo0"]);
    }

    #[test]
    fn interfaces_localhost() {
        let ifaces = get_interfaces("localhost");
        assert_eq!(ifaces, vec!["lo0"]);
    }

    #[test]
    fn interfaces_specific_ip() {
        let ifaces = get_interfaces("192.168.1.100");
        assert_eq!(ifaces, vec!["lo0"]);
    }

    #[test]
    fn interfaces_all_ipv4_includes_loopback() {
        let ifaces = get_interfaces("0.0.0.0");
        assert!(ifaces.contains(&"lo0".to_string()));
    }

    #[test]
    fn interfaces_all_ipv6_includes_loopback() {
        let ifaces = get_interfaces("::");
        assert!(ifaces.contains(&"lo0".to_string()));
    }

    #[test]
    fn interfaces_no_duplicates() {
        let ifaces = get_interfaces("0.0.0.0");
        let mut seen = std::collections::HashSet::new();
        for iface in &ifaces {
            assert!(seen.insert(iface), "Duplicate interface: {iface}");
        }
    }

    #[test]
    fn default_interface_returns_valid_name() {
        if let Some(iface) = get_default_interface() {
            assert!(!iface.is_empty());
            assert!(
                iface.chars().all(|c| c.is_alphanumeric()),
                "Unexpected interface name: {iface}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Cleanup hint
    // -----------------------------------------------------------------------

    #[test]
    fn cleanup_hint_returns_pfctl_command() {
        let hints = cleanup_hint(&[(443, 8443)]);
        assert_eq!(hints.len(), 1);
        assert!(hints[0].contains("pfctl"));
        assert!(hints[0].contains(PF_ANCHOR));
    }

    // -----------------------------------------------------------------------
    // Integration (requires sudo)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "requires sudo"]
    fn integration_setup_and_cleanup() {
        use std::process::Command;
        setup(&[(443, 8443), (80, 8080)], "127.0.0.1").expect("setup");

        let output = Command::new("sudo")
            .args(["pfctl", "-a", PF_ANCHOR, "-s", "nat"])
            .output()
            .expect("pfctl -s nat");
        let rules = String::from_utf8_lossy(&output.stdout);
        assert!(rules.contains("8443"), "Expected 8443: {rules}");
        assert!(rules.contains("8080"), "Expected 8080: {rules}");

        cleanup(&[(443, 8443), (80, 8080)]).expect("cleanup");

        let output = Command::new("sudo")
            .args(["pfctl", "-a", PF_ANCHOR, "-s", "nat"])
            .output()
            .expect("pfctl after cleanup");
        let rules = String::from_utf8_lossy(&output.stdout);
        assert!(!rules.contains("8443"), "Should be cleaned: {rules}");
    }

    #[test]
    #[ignore = "requires sudo"]
    fn integration_all_interfaces() {
        setup(&[(443, 8443)], "0.0.0.0").expect("setup");

        let output = Command::new("sudo")
            .args(["pfctl", "-a", PF_ANCHOR, "-s", "nat"])
            .output()
            .expect("pfctl -s nat");
        let rules = String::from_utf8_lossy(&output.stdout);
        assert!(rules.contains("lo0"), "Expected lo0: {rules}");
        assert!(rules.contains("8443"), "Expected 8443: {rules}");

        cleanup(&[(443, 8443)]).expect("cleanup");
    }
}
