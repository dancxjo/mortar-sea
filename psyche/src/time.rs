use chrono::{DateTime, Local, SecondsFormat, Utc};

/// Returns the current UTC time.
pub fn now() -> DateTime<Utc> {
    Utc::now()
}

/// Formats a UTC timestamp as local RFC 3339 / ISO 8601 with an explicit offset.
pub fn local_iso(timestamp: DateTime<Utc>) -> String {
    timestamp
        .with_timezone(&Local)
        .to_rfc3339_opts(SecondsFormat::Millis, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_iso_includes_explicit_offset() {
        let timestamp = DateTime::parse_from_rfc3339("2026-06-04T12:34:56.789Z")
            .expect("valid timestamp")
            .with_timezone(&Utc);

        let formatted = local_iso(timestamp);

        assert!(!formatted.ends_with('Z'));
        let offset = &formatted[formatted.len() - 6..];
        assert!(offset.starts_with('+') || offset.starts_with('-'));
        assert_eq!(&offset[3..4], ":");
        let parsed = DateTime::parse_from_rfc3339(&formatted).expect("valid local ISO timestamp");
        assert_eq!(parsed.with_timezone(&Utc), timestamp);
    }
}
