use chrono::{DateTime, Utc};

/// Returns the current UTC time.
pub fn now() -> DateTime<Utc> {
    Utc::now()
}
