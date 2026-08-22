#[cfg(test)]
pub mod mock;
pub mod ollama;
pub mod openai_compat;
pub mod provider;

pub use ollama::OllamaProvider;
pub use openai_compat::OpenAiCompatProvider;
#[allow(unused_imports)]
pub use provider::{ChatOptions, ChunkStream, LlmProvider, LlmStreamChunk, ModelInfo};
