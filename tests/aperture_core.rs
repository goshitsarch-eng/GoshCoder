#![allow(dead_code)]

// Compile the Aperture core as a standalone integration-test module so its
// pure routing, cache, and metadata logic is exercised without the runtime
// around it. The production wiring lives in src/catalog.rs (dynamic layer)
// and src/runtime.rs (session start).
#[path = "../src/llm.rs"]
mod llm;

#[path = "../src/aperture.rs"]
mod aperture;
