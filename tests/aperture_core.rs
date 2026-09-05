#![allow(dead_code)]

// Compile the standalone Aperture port as an integration-test module without
// wiring it into src/main.rs. The production integration is intentionally left
// for the surrounding Rust runtime migration.
#[path = "../src/llm.rs"]
mod llm;

#[path = "../src/aperture.rs"]
mod aperture;
