pub mod app;
mod asr;
mod face_detection;
mod ingestion;
mod llm_scheduler;
mod location;
mod memory;
mod messages;
mod realtime_experience;
mod vision;
mod voice;

pub use app::run;
