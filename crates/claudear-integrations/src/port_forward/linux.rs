//! Linux port forwarding via `iptables` NAT table.
//!
//! Two rules per redirect:
//! - `PREROUTING`: catches external/network traffic arriving on the port.
//! - `OUTPUT -o lo`: catches local loopback traffic (e.g. `curl localhost:443`).
//!
//! Rules are tagged with the `claudear-port-forward` comment for identification.

use std::process::{Command, Stdio};

/// iptables comment tag used to identify our rules for clean removal.
pub const IPTABLES_COMMENT: &str = "claudear-port-forward";

/// Apply iptables NAT REDIRECT rules.
pub fn setup(redirects: &[(u16, u16)], _bind_address: &str) -> Result<(), String> {
    for &(from_port, to_port) in redirects {
        let args = build_prerouting_args("-A", from_port, to_port);
        run_iptables_interactive(&args)?;

        let args = build_output_lo_args("-A", from_port, to_port);
        run_iptables_interactive(&args)?;
    }

    tracing::info!("Linux iptables port forwarding active");
    Ok(())
}

/// Remove our iptables rules. Uses `sudo -n` (non-interactive) so cleanup
/// never blocks on a password prompt during shutdown.
pub fn cleanup(redirects: &[(u16, u16)]) -> Result<(), String> {
    tracing::info!("Cleaning up Linux iptables port forwarding rules");

    let mut errors = Vec::new();

    for &(from_port, to_port) in redirects {
        let args = build_prerouting_args("-D", from_port, to_port);
        if let Err(e) = run_iptables_noninteractive(&args) {
            errors.push(e);
        }

        let args = build_output_lo_args("-D", from_port, to_port);
        if let Err(e) = run_iptables_noninteractive(&args) {
            errors.push(e);
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("; "))
    }
}

/// Return manual cleanup commands the user can run.
pub fn cleanup_hint(redirects: &[(u16, u16)]) -> Vec<String> {
    let mut hints = Vec::new();
    for (from, to) in redirects {
        hints.push(format!(
            "sudo iptables -t nat -D PREROUTING -p tcp --dport {from} \
             -j REDIRECT --to-port {to} -m comment --comment {IPTABLES_COMMENT}"
        ));
        hints.push(format!(
            "sudo iptables -t nat -D OUTPUT -o lo -p tcp --dport {from} \
             -j REDIRECT --to-port {to} -m comment --comment {IPTABLES_COMMENT}"
        ));
    }
    hints
}

// ---------------------------------------------------------------------------
// Pure helpers (testable without sudo / iptables)
// ---------------------------------------------------------------------------

/// Build iptables arguments for a PREROUTING REDIRECT rule.
pub fn build_prerouting_args(action: &str, from_port: u16, to_port: u16) -> Vec<String> {
    vec![
        "-t".into(),
        "nat".into(),
        action.into(),
        "PREROUTING".into(),
        "-p".into(),
        "tcp".into(),
        "--dport".into(),
        from_port.to_string(),
        "-j".into(),
        "REDIRECT".into(),
        "--to-port".into(),
        to_port.to_string(),
        "-m".into(),
        "comment".into(),
        "--comment".into(),
        IPTABLES_COMMENT.into(),
    ]
}

/// Build iptables arguments for an OUTPUT loopback REDIRECT rule.
pub fn build_output_lo_args(action: &str, from_port: u16, to_port: u16) -> Vec<String> {
    vec![
        "-t".into(),
        "nat".into(),
        action.into(),
        "OUTPUT".into(),
        "-o".into(),
        "lo".into(),
        "-p".into(),
        "tcp".into(),
        "--dport".into(),
        from_port.to_string(),
        "-j".into(),
        "REDIRECT".into(),
        "--to-port".into(),
        to_port.to_string(),
        "-m".into(),
        "comment".into(),
        "--comment".into(),
        IPTABLES_COMMENT.into(),
    ]
}

// ---------------------------------------------------------------------------
// Command runners
// ---------------------------------------------------------------------------

/// Run an iptables command with sudo (interactive — for setup).
fn run_iptables_interactive(args: &[String]) -> Result<(), String> {
    let mut cmd_args = vec!["iptables".to_string()];
    cmd_args.extend_from_slice(args);

    let output = Command::new("sudo")
        .args(&cmd_args)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to run sudo iptables: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("iptables failed: {stderr}"));
    }

    Ok(())
}

/// Run an iptables command with `sudo -n` (non-interactive — for cleanup).
fn run_iptables_noninteractive(args: &[String]) -> Result<(), String> {
    let mut cmd_args = vec!["-n".to_string(), "iptables".to_string()];
    cmd_args.extend_from_slice(args);

    let output = Command::new("sudo")
        .args(&cmd_args)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .map_err(|e| format!("Failed to run iptables cleanup: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("iptables cleanup failed: {stderr}"));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // PREROUTING arg generation
    // -----------------------------------------------------------------------

    #[test]
    fn prerouting_append() {
        let args = build_prerouting_args("-A", 443, 8443);
        assert_eq!(
            args,
            vec![
                "-t",
                "nat",
                "-A",
                "PREROUTING",
                "-p",
                "tcp",
                "--dport",
                "443",
                "-j",
                "REDIRECT",
                "--to-port",
                "8443",
                "-m",
                "comment",
                "--comment",
                IPTABLES_COMMENT,
            ]
        );
    }

    #[test]
    fn prerouting_delete() {
        let args = build_prerouting_args("-D", 80, 8080);
        assert_eq!(args[2], "-D");
        assert_eq!(args[7], "80");
        assert_eq!(args[11], "8080");
    }

    #[test]
    fn prerouting_uses_nat_table() {
        let args = build_prerouting_args("-A", 443, 8443);
        assert_eq!(&args[0..2], &["-t", "nat"]);
    }

    #[test]
    fn prerouting_no_interface_flag() {
        let args = build_prerouting_args("-A", 443, 8443);
        assert!(!args.contains(&"-o".to_string()));
        assert!(!args.contains(&"-i".to_string()));
    }

    // -----------------------------------------------------------------------
    // OUTPUT loopback arg generation
    // -----------------------------------------------------------------------

    #[test]
    fn output_lo_append() {
        let args = build_output_lo_args("-A", 443, 8443);
        assert_eq!(
            args,
            vec![
                "-t",
                "nat",
                "-A",
                "OUTPUT",
                "-o",
                "lo",
                "-p",
                "tcp",
                "--dport",
                "443",
                "-j",
                "REDIRECT",
                "--to-port",
                "8443",
                "-m",
                "comment",
                "--comment",
                IPTABLES_COMMENT,
            ]
        );
    }

    #[test]
    fn output_lo_delete() {
        let args = build_output_lo_args("-D", 80, 8080);
        assert_eq!(args[2], "-D");
        assert_eq!(args[3], "OUTPUT");
        assert_eq!(args[5], "lo");
        assert_eq!(args[9], "80");
        assert_eq!(args[13], "8080");
    }

    #[test]
    fn output_lo_targets_loopback() {
        let args = build_output_lo_args("-A", 443, 8443);
        let o_idx = args.iter().position(|a| a == "-o").unwrap();
        assert_eq!(args[o_idx + 1], "lo");
    }

    // -----------------------------------------------------------------------
    // Shared properties
    // -----------------------------------------------------------------------

    #[test]
    fn both_include_comment() {
        let pre = build_prerouting_args("-A", 443, 8443);
        let out = build_output_lo_args("-A", 443, 8443);
        for args in [&pre, &out] {
            let idx = args.iter().position(|a| a == "--comment").unwrap();
            assert_eq!(args[idx + 1], IPTABLES_COMMENT);
        }
    }

    #[test]
    fn both_use_tcp() {
        let pre = build_prerouting_args("-A", 443, 8443);
        let out = build_output_lo_args("-A", 443, 8443);
        for args in [&pre, &out] {
            let idx = args.iter().position(|a| a == "-p").unwrap();
            assert_eq!(args[idx + 1], "tcp");
        }
    }

    #[test]
    fn append_and_delete_are_symmetric() {
        for builder in [build_prerouting_args, build_output_lo_args] {
            let append = builder("-A", 443, 8443);
            let delete = builder("-D", 443, 8443);
            assert_eq!(append.len(), delete.len());
            for (i, (a, d)) in append.iter().zip(delete.iter()).enumerate() {
                if i == 2 {
                    assert_eq!(a, "-A");
                    assert_eq!(d, "-D");
                } else {
                    assert_eq!(a, d, "Mismatch at index {i}");
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Edge case ports
    // -----------------------------------------------------------------------

    #[test]
    fn port_zero() {
        let args = build_prerouting_args("-A", 0, 8000);
        assert!(args.contains(&"0".to_string()));
        assert!(args.contains(&"8000".to_string()));
    }

    #[test]
    fn max_port() {
        let args = build_prerouting_args("-A", 1023, 65535);
        assert!(args.contains(&"1023".to_string()));
        assert!(args.contains(&"65535".to_string()));
    }

    // -----------------------------------------------------------------------
    // Cleanup hint
    // -----------------------------------------------------------------------

    #[test]
    fn cleanup_hint_single_redirect() {
        let hints = cleanup_hint(&[(443, 8443)]);
        assert_eq!(hints.len(), 2); // PREROUTING + OUTPUT
        assert!(hints[0].contains("PREROUTING"));
        assert!(hints[0].contains("443"));
        assert!(hints[0].contains("8443"));
        assert!(hints[1].contains("OUTPUT"));
        assert!(hints[1].contains("443"));
        assert!(hints[1].contains("8443"));
    }

    #[test]
    fn cleanup_hint_multiple_redirects() {
        let hints = cleanup_hint(&[(443, 8443), (80, 8080)]);
        assert_eq!(hints.len(), 4); // 2 per redirect
    }

    #[test]
    fn cleanup_hint_includes_comment() {
        let hints = cleanup_hint(&[(443, 8443)]);
        for hint in &hints {
            assert!(hint.contains(IPTABLES_COMMENT));
        }
    }

    #[test]
    fn cleanup_hint_empty() {
        assert!(cleanup_hint(&[]).is_empty());
    }

    // -----------------------------------------------------------------------
    // Integration (requires sudo + iptables)
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "requires sudo and iptables"]
    fn integration_setup_and_cleanup() {
        setup(&[(443, 8443), (80, 8080)], "0.0.0.0").expect("setup");

        let output = Command::new("sudo")
            .args(["iptables", "-t", "nat", "-L", "PREROUTING", "-n"])
            .output()
            .expect("list PREROUTING");
        let rules = String::from_utf8_lossy(&output.stdout);
        assert!(rules.contains("8443"), "Expected 8443: {rules}");
        assert!(rules.contains("8080"), "Expected 8080: {rules}");

        cleanup(&[(443, 8443), (80, 8080)]).expect("cleanup");

        let output = Command::new("sudo")
            .args(["iptables", "-t", "nat", "-L", "PREROUTING", "-n"])
            .output()
            .expect("list after cleanup");
        let rules = String::from_utf8_lossy(&output.stdout);
        assert!(!rules.contains("8443"), "Should be cleaned: {rules}");
    }

    #[test]
    #[ignore = "requires sudo and iptables"]
    fn integration_idempotent_cleanup() {
        setup(&[(443, 8443)], "0.0.0.0").expect("setup");
        cleanup(&[(443, 8443)]).expect("first cleanup");
        // Second cleanup should not panic (rules already gone).
        let _ = cleanup(&[(443, 8443)]);
    }
}
