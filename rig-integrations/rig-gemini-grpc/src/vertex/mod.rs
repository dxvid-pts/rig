//! Vertex AI (Google Cloud AI Platform) gRPC integration for Rig.
//!
//! This module targets the `google.cloud.aiplatform.v1.PredictionService`
//! Generative AI APIs, providing:
//! - `GenerateContent` (unary)
//! - `StreamGenerateContent` (server streaming)
//! - `EmbedContent`

pub mod client;
pub mod completion;
pub mod embedding;
pub mod streaming;

pub use client::{VertexClient, VertexClientBuilder, VertexGrpcEndpoint};

// Re-export Vertex proto types under this module to avoid collisions with the
// Google AI Studio Gemini proto types re-exported at the crate root.
pub use crate::vertex_proto::{
    Candidate, Content, EmbedContentRequest, EmbedContentResponse, FileData, FunctionCall,
    FunctionDeclaration, FunctionResponse, GenerateContentRequest, GenerateContentResponse,
    GenerationConfig, Part, Tool, ToolConfig, UsageMetadata,
    prediction_service_client::PredictionServiceClient,
};
