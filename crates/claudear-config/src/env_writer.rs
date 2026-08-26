//! Utility for writing to .env files.

use claudear_core::error::{Error, Result};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Updates or appends key-value pairs in a .env file.
///
/// This function:
/// - Creates the file if it doesn't exist
/// - Updates existing keys with new values
/// - Appends new keys at the end
/// - Preserves comments and formatting
pub fn update_env_file(path: &Path, updates: &HashMap<String, String>) -> Result<()> {
    let content = if path.exists() {
        fs::read_to_string(path)
            .map_err(|e| Error::config(format!("Failed to read .env file at {:?}: {}", path, e)))?
    } else {
        String::new()
    };

    let updated_content = update_env_content(&content, updates);

    fs::write(path, updated_content)
        .map_err(|e| Error::config(format!("Failed to write .env file at {:?}: {}", path, e)))?;

    // Set restrictive permissions on the .env file (owner read/write only)
    // since it may contain secrets like webhook secrets and API tokens.
    claudear_core::platform::set_file_permissions_secure(path).ok();

    Ok(())
}

/// Updates .env content string with new key-value pairs.
///
/// Returns the updated content.
fn update_env_content(content: &str, updates: &HashMap<String, String>) -> String {
    let mut result_lines: Vec<String> = Vec::new();
    let mut updated_keys: std::collections::HashSet<String> = std::collections::HashSet::new();

    for line in content.lines() {
        let trimmed = line.trim();

        // Preserve empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            result_lines.push(line.to_string());
            continue;
        }

        // Parse key=value (handle various formats)
        if let Some((key, _)) = parse_env_line(trimmed) {
            if let Some(new_value) = updates.get(&key) {
                // Update existing key
                result_lines.push(format!("{}={}", key, quote_value(new_value)));
                updated_keys.insert(key);
            } else {
                // Keep original line
                result_lines.push(line.to_string());
            }
        } else {
            // Keep unrecognized lines
            result_lines.push(line.to_string());
        }
    }

    // Append new keys that weren't in the original file
    let mut new_keys: Vec<_> = updates
        .iter()
        .filter(|(k, _)| !updated_keys.contains(*k))
        .collect();
    new_keys.sort_by_key(|(a, _)| *a); // Sort for deterministic output

    if !new_keys.is_empty() {
        // Add a blank line before new keys if the file isn't empty and doesn't end with blank line
        if !result_lines.is_empty() {
            let last_line = result_lines.last().unwrap();
            if !last_line.trim().is_empty() {
                result_lines.push(String::new());
            }
        }

        // Add comment for auto-generated section
        result_lines.push("# Auto-configured webhook secrets".to_string());

        for (key, value) in new_keys {
            result_lines.push(format!("{}={}", key, quote_value(value)));
        }
    }

    // Ensure file ends with newline
    let mut result = result_lines.join("\n");
    if !result.is_empty() && !result.ends_with('\n') {
        result.push('\n');
    }

    result
}

/// Parse a line into key and value.
fn parse_env_line(line: &str) -> Option<(String, String)> {
    let line = line.trim();

    // Skip comments and empty lines
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    // Find the first '=' sign
    let eq_pos = line.find('=')?;
    let key = line[..eq_pos].trim().to_string();
    let value = line[eq_pos + 1..].trim();

    // Remove surrounding quotes if present
    let value = if (value.starts_with('"') && value.ends_with('"'))
        || (value.starts_with('\'') && value.ends_with('\''))
    {
        value[1..value.len() - 1].to_string()
    } else {
        value.to_string()
    };

    Some((key, value))
}

/// Quote a value if it contains special characters.
fn quote_value(value: &str) -> String {
    // Quote if contains spaces, quotes, or special shell characters
    if value.contains(' ')
        || value.contains('"')
        || value.contains('\'')
        || value.contains('#')
        || value.contains('$')
        || value.contains('\n')
        || value.contains('\\')
    {
        // Use double quotes and escape internal quotes
        let escaped = value.replace('\\', "\\\\").replace('"', "\\\"");
        format!("\"{}\"", escaped)
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tempfile::TempDir;

    #[test]
    fn test_parse_env_line_simple() {
        let (key, value) = parse_env_line("KEY=value").unwrap();
        assert_eq!(key, "KEY");
        assert_eq!(value, "value");
    }

    #[test]
    fn test_parse_env_line_with_spaces() {
        let (key, value) = parse_env_line("  KEY  =  value  ").unwrap();
        assert_eq!(key, "KEY");
        assert_eq!(value, "value");
    }

    #[test]
    fn test_parse_env_line_double_quoted() {
        let (key, value) = parse_env_line("KEY=\"quoted value\"").unwrap();
        assert_eq!(key, "KEY");
        assert_eq!(value, "quoted value");
    }

    #[test]
    fn test_parse_env_line_single_quoted() {
        let (key, value) = parse_env_line("KEY='quoted value'").unwrap();
        assert_eq!(key, "KEY");
        assert_eq!(value, "quoted value");
    }

    #[test]
    fn test_parse_env_line_empty_value() {
        let (key, value) = parse_env_line("KEY=").unwrap();
        assert_eq!(key, "KEY");
        assert_eq!(value, "");
    }

    #[test]
    fn test_parse_env_line_comment() {
        assert!(parse_env_line("# comment").is_none());
    }

    #[test]
    fn test_parse_env_line_empty() {
        assert!(parse_env_line("").is_none());
        assert!(parse_env_line("   ").is_none());
    }

    #[test]
    fn test_parse_env_line_value_with_equals() {
        let (key, value) = parse_env_line("KEY=value=with=equals").unwrap();
        assert_eq!(key, "KEY");
        assert_eq!(value, "value=with=equals");
    }

    #[test]
    fn test_quote_value_simple() {
        assert_eq!(quote_value("simple"), "simple");
    }

    #[test]
    fn test_quote_value_with_spaces() {
        assert_eq!(quote_value("with spaces"), "\"with spaces\"");
    }

    #[test]
    fn test_quote_value_with_quotes() {
        assert_eq!(quote_value("with\"quote"), "\"with\\\"quote\"");
    }

    #[test]
    fn test_quote_value_with_hash() {
        assert_eq!(quote_value("with#hash"), "\"with#hash\"");
    }

    #[test]
    fn test_update_env_content_new_key() {
        let content = "EXISTING=value\n";
        let mut updates = HashMap::new();
        updates.insert("NEW_KEY".to_string(), "new_value".to_string());

        let result = update_env_content(content, &updates);
        assert!(result.contains("EXISTING=value"));
        assert!(result.contains("NEW_KEY=new_value"));
        assert!(result.contains("# Auto-configured webhook secrets"));
    }

    #[test]
    fn test_update_env_content_update_existing() {
        let content = "KEY=old_value\n";
        let mut updates = HashMap::new();
        updates.insert("KEY".to_string(), "new_value".to_string());

        let result = update_env_content(content, &updates);
        assert!(result.contains("KEY=new_value"));
        assert!(!result.contains("old_value"));
    }

    #[test]
    fn test_update_env_content_preserve_comments() {
        let content = "# This is a comment\nKEY=value\n";
        let mut updates = HashMap::new();
        updates.insert("KEY".to_string(), "new_value".to_string());

        let result = update_env_content(content, &updates);
        assert!(result.contains("# This is a comment"));
        assert!(result.contains("KEY=new_value"));
    }

    #[test]
    fn test_update_env_content_preserve_empty_lines() {
        let content = "KEY1=value1\n\nKEY2=value2\n";
        let mut updates = HashMap::new();
        updates.insert("KEY1".to_string(), "new1".to_string());

        let result = update_env_content(content, &updates);
        assert!(result.contains("KEY1=new1"));
        assert!(result.contains("\n\n")); // Empty line preserved
    }

    #[test]
    fn test_update_env_content_empty_file() {
        let content = "";
        let mut updates = HashMap::new();
        updates.insert("NEW_KEY".to_string(), "value".to_string());

        let result = update_env_content(content, &updates);
        assert!(result.contains("NEW_KEY=value"));
    }

    #[test]
    fn test_update_env_content_multiple_updates() {
        let content = "KEY1=old1\nKEY2=old2\n";
        let mut updates = HashMap::new();
        updates.insert("KEY1".to_string(), "new1".to_string());
        updates.insert("KEY3".to_string(), "new3".to_string());

        let result = update_env_content(content, &updates);
        assert!(result.contains("KEY1=new1"));
        assert!(result.contains("KEY2=old2"));
        assert!(result.contains("KEY3=new3"));
    }

    #[test]
    fn test_update_env_file_creates_new() {
        let temp_dir = TempDir::new().unwrap();
        let env_path = temp_dir.path().join(".env");

        let mut updates = HashMap::new();
        updates.insert("KEY".to_string(), "value".to_string());

        update_env_file(&env_path, &updates).unwrap();

        let content = fs::read_to_string(&env_path).unwrap();
        assert!(content.contains("KEY=value"));
    }

    #[test]
    fn test_update_env_file_updates_existing() {
        let temp_dir = TempDir::new().unwrap();
        let env_path = temp_dir.path().join(".env");

        // Create initial file
        fs::write(&env_path, "EXISTING=old\n").unwrap();

        let mut updates = HashMap::new();
        updates.insert("EXISTING".to_string(), "new".to_string());
        updates.insert("NEW_KEY".to_string(), "value".to_string());

        update_env_file(&env_path, &updates).unwrap();

        let content = fs::read_to_string(&env_path).unwrap();
        assert!(content.contains("EXISTING=new"));
        assert!(content.contains("NEW_KEY=value"));
    }

    #[test]
    fn test_update_env_content_ends_with_newline() {
        let content = "KEY=value";
        let updates = HashMap::new();

        let result = update_env_content(content, &updates);
        assert!(result.ends_with('\n'));
    }

    #[test]
    fn test_parse_env_line_no_equals() {
        assert!(parse_env_line("NOEQUALS").is_none());
    }

    #[test]
    fn test_quote_value_with_backslash() {
        let result = quote_value("path\\to\\file");
        assert_eq!(result, "\"path\\\\to\\\\file\"");
    }

    #[test]
    fn test_quote_value_with_dollar() {
        assert_eq!(quote_value("value$var"), "\"value$var\"");
    }

    #[test]
    fn test_update_env_content_preserves_order() {
        let content = "FIRST=1\nSECOND=2\nTHIRD=3\n";
        let mut updates = HashMap::new();
        updates.insert("SECOND".to_string(), "updated".to_string());

        let result = update_env_content(content, &updates);
        let first_pos = result.find("FIRST").unwrap();
        let second_pos = result.find("SECOND").unwrap();
        let third_pos = result.find("THIRD").unwrap();

        assert!(first_pos < second_pos);
        assert!(second_pos < third_pos);
    }
}
