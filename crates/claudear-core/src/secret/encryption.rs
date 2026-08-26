//! AES-256-GCM encryption for secret values at rest.
//!
//! Secrets are encrypted using the format: `ENC[v1:<base64(nonce‖ciphertext‖tag)>]`
//! - Each field is encrypted with its own random 12-byte nonce
//! - Plaintext values are still accepted (backwards-compatible)
//! - Warnings are logged when plaintext secrets are detected in config

use aes_gcm::{
    aead::{Aead, KeyInit, OsRng},
    AeadCore, Aes256Gcm, Nonce,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use std::path::{Path, PathBuf};
use zeroize::Zeroize;

/// The encryption format prefix.
const ENC_PREFIX: &str = "ENC[v1:";
/// The encryption format suffix.
const ENC_SUFFIX: &str = "]";
/// AES-256-GCM nonce size in bytes.
const NONCE_SIZE: usize = 12;

/// Master key for encrypting/decrypting secrets.
///
/// Wraps a 32-byte AES-256 key and zeroizes on drop.
pub struct MasterKey {
    key: [u8; 32],
}

impl MasterKey {
    /// Create a master key from raw bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self { key: bytes }
    }

    /// Create a master key from a hex-encoded string.
    pub fn from_hex(hex: &str) -> Result<Self, String> {
        let bytes = hex::decode(hex.trim())
            .map_err(|e| format!("Invalid hex-encoded master key: {}", e))?;
        if bytes.len() != 32 {
            return Err(format!("Master key must be 32 bytes, got {}", bytes.len()));
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(&bytes);
        Ok(Self { key })
    }

    /// Generate a new random master key.
    pub fn generate() -> Self {
        let key = Aes256Gcm::generate_key(OsRng);
        let mut bytes = [0u8; 32];
        bytes.copy_from_slice(&key);
        Self { key: bytes }
    }

    /// Encode the master key as a hex string.
    pub fn to_hex(&self) -> String {
        hex::encode(self.key)
    }

    /// Get the raw key bytes.
    fn as_bytes(&self) -> &[u8; 32] {
        &self.key
    }
}

impl std::fmt::Debug for MasterKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MasterKey([REDACTED])")
    }
}

impl Drop for MasterKey {
    fn drop(&mut self) {
        self.key.zeroize();
    }
}

/// Check if a string value is in encrypted format.
pub fn is_encrypted(value: &str) -> bool {
    value.starts_with(ENC_PREFIX) && value.ends_with(ENC_SUFFIX)
}

/// Encrypt a plaintext secret value.
///
/// Returns the encrypted string in `ENC[v1:<base64>]` format.
pub fn encrypt_value(plaintext: &str, master_key: &MasterKey) -> Result<String, String> {
    let cipher = Aes256Gcm::new_from_slice(master_key.as_bytes())
        .map_err(|e| format!("Failed to create cipher: {}", e))?;

    let nonce = Aes256Gcm::generate_nonce(&mut OsRng);

    let ciphertext = cipher
        .encrypt(&nonce, plaintext.as_bytes())
        .map_err(|e| format!("Encryption failed: {}", e))?;

    // Combine nonce + ciphertext (tag is appended by AES-GCM)
    let mut combined = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
    combined.extend_from_slice(&nonce);
    combined.extend_from_slice(&ciphertext);

    let encoded = BASE64.encode(&combined);
    Ok(format!("{}{}{}", ENC_PREFIX, encoded, ENC_SUFFIX))
}

/// Decrypt an encrypted secret value.
///
/// Accepts either `ENC[v1:<base64>]` format or returns plaintext as-is.
pub fn decrypt_value(value: &str, master_key: &MasterKey) -> Result<String, String> {
    if !is_encrypted(value) {
        // Plaintext - return as-is (backwards compatible)
        return Ok(value.to_string());
    }

    // Strip prefix and suffix
    let encoded = &value[ENC_PREFIX.len()..value.len() - ENC_SUFFIX.len()];

    let combined = BASE64
        .decode(encoded)
        .map_err(|e| format!("Invalid base64 in encrypted value: {}", e))?;

    if combined.len() < NONCE_SIZE + 16 {
        // 16 = minimum tag size
        return Err("Encrypted value too short".to_string());
    }

    let (nonce_bytes, ciphertext) = combined.split_at(NONCE_SIZE);
    let nonce = Nonce::from_slice(nonce_bytes);

    let cipher = Aes256Gcm::new_from_slice(master_key.as_bytes())
        .map_err(|e| format!("Failed to create cipher: {}", e))?;

    let plaintext = cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| "Decryption failed: invalid key or corrupted data".to_string())?;

    String::from_utf8(plaintext).map_err(|e| format!("Decrypted value is not valid UTF-8: {}", e))
}

/// Default master key file path.
pub fn default_key_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".claudear")
        .join("master.key")
}

/// Load the master key from configured sources.
///
/// Checks in order:
/// 1. `CLAUDEAR_MASTER_KEY` env var (hex-encoded 32 bytes)
/// 2. `CLAUDEAR_MASTER_KEY_FILE` env var (path to key file)
/// 3. `~/.claudear/master.key` file
///
/// Returns `None` if no key is configured (plaintext mode).
pub fn load_master_key() -> Result<Option<MasterKey>, String> {
    // 1. Check env var (hex-encoded)
    if let Ok(hex_key) = std::env::var("CLAUDEAR_MASTER_KEY") {
        if !hex_key.is_empty() {
            return MasterKey::from_hex(&hex_key).map(Some);
        }
    }

    // 2. Check env var pointing to key file
    if let Ok(key_path) = std::env::var("CLAUDEAR_MASTER_KEY_FILE") {
        if !key_path.is_empty() {
            return load_key_from_file(Path::new(&key_path)).map(Some);
        }
    }

    // 3. Check default key file
    let default_path = default_key_path();
    if default_path.exists() {
        return load_key_from_file(&default_path).map(Some);
    }

    Ok(None)
}

/// Load a master key from a file (hex-encoded content).
fn load_key_from_file(path: &Path) -> Result<MasterKey, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read master key file '{}': {}", path.display(), e))?;
    MasterKey::from_hex(content.trim())
}

/// Write a master key to a file with restrictive permissions.
pub fn write_key_file(path: &Path, key: &MasterKey) -> Result<(), String> {
    // Create parent directory
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory '{}': {}", parent.display(), e))?;
    }

    std::fs::write(path, key.to_hex())
        .map_err(|e| format!("Failed to write key file '{}': {}", path.display(), e))?;

    // Set restrictive permissions (owner read/write only)
    crate::platform::set_file_permissions_secure(path)
        .map_err(|e| format!("Failed to set key file permissions: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = MasterKey::generate();
        let plaintext = "my-secret-token-ghp_abc123";

        let encrypted = encrypt_value(plaintext, &key).unwrap();
        assert!(is_encrypted(&encrypted));
        assert!(encrypted.starts_with(ENC_PREFIX));
        assert!(encrypted.ends_with(ENC_SUFFIX));
        assert!(!encrypted.contains(plaintext));

        let decrypted = decrypt_value(&encrypted, &key).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_decrypt_plaintext_passthrough() {
        let key = MasterKey::generate();
        let plaintext = "just-a-plain-token";

        let result = decrypt_value(plaintext, &key).unwrap();
        assert_eq!(result, plaintext);
    }

    #[test]
    fn test_is_encrypted() {
        assert!(is_encrypted("ENC[v1:abc123==]"));
        assert!(!is_encrypted("plaintext"));
        assert!(!is_encrypted("ENC[v1:missing-suffix"));
        assert!(!is_encrypted("wrong[v1:abc]"));
    }

    #[test]
    fn test_wrong_key_fails() {
        let key1 = MasterKey::generate();
        let key2 = MasterKey::generate();

        let encrypted = encrypt_value("secret", &key1).unwrap();
        let result = decrypt_value(&encrypted, &key2);
        assert!(result.is_err());
    }

    #[test]
    fn test_master_key_from_hex() {
        let key = MasterKey::generate();
        let hex = key.to_hex();
        let restored = MasterKey::from_hex(&hex).unwrap();
        assert_eq!(key.as_bytes(), restored.as_bytes());
    }

    #[test]
    fn test_master_key_from_hex_invalid_length() {
        let result = MasterKey::from_hex("abcdef");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("32 bytes"));
    }

    #[test]
    fn test_master_key_from_hex_invalid_chars() {
        let result =
            MasterKey::from_hex("not_valid_hex_string_of_64_chars_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx");
        assert!(result.is_err());
    }

    #[test]
    fn test_each_encryption_unique() {
        let key = MasterKey::generate();
        let plaintext = "same-secret";

        let enc1 = encrypt_value(plaintext, &key).unwrap();
        let enc2 = encrypt_value(plaintext, &key).unwrap();

        // Different nonces produce different ciphertext
        assert_ne!(enc1, enc2);

        // Both decrypt to the same plaintext
        assert_eq!(decrypt_value(&enc1, &key).unwrap(), plaintext);
        assert_eq!(decrypt_value(&enc2, &key).unwrap(), plaintext);
    }

    #[test]
    fn test_empty_string_encrypt_decrypt() {
        let key = MasterKey::generate();
        let encrypted = encrypt_value("", &key).unwrap();
        let decrypted = decrypt_value(&encrypted, &key).unwrap();
        assert_eq!(decrypted, "");
    }

    #[test]
    fn test_write_and_read_key_file() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let key_path = temp_dir.path().join("test.key");

        let original = MasterKey::generate();
        write_key_file(&key_path, &original).unwrap();

        let loaded = load_key_from_file(&key_path).unwrap();
        assert_eq!(original.as_bytes(), loaded.as_bytes());
    }
}
