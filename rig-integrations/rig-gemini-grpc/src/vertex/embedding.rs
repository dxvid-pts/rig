// ================================================================
//! Vertex AI gRPC Embedding Integration
// ================================================================

/// `text-embedding-004` embedding model
pub const EMBEDDING_004: &str = "text-embedding-004";

use rig::embeddings::{self, EmbeddingError};
use tracing::{Instrument, info_span};

use crate::vertex::client::VertexClient;
use crate::vertex_proto as proto;

#[derive(Clone, Debug)]
pub struct EmbeddingModel {
    client: VertexClient,
    model: String,
    ndims: usize,
}

impl EmbeddingModel {
    pub fn new(client: VertexClient, model: impl Into<String>, dims: Option<usize>) -> Self {
        Self {
            client,
            model: model.into(),
            ndims: dims.unwrap_or(768),
        }
    }
}

impl embeddings::EmbeddingModel for EmbeddingModel {
    const MAX_DOCUMENTS: usize = 100;

    type Client = VertexClient;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        Self::new(client.clone(), model, dims)
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String> + rig::wasm_compat::WasmCompatSend,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents_vec: Vec<String> = documents.into_iter().collect();

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::embeddings",
                "embed_content",
                gen_ai.operation.name = "embed_content",
                gen_ai.provider.name = "gcp.vertexai",
                gen_ai.request.model = &self.model,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.response.model_name = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        let client = self.client.clone();
        let model = self.model.clone();
        let ndims = self.ndims;

        let async_block = async move {
            let mut embeddings_out = Vec::new();
            let mut grpc_client = client.grpc_client();

            let mut total_usage = rig::completion::Usage::default();

            for doc in documents_vec {
                let request = proto::EmbedContentRequest {
                    model: Some(client.resolve_model_path(&model)),
                    content: Some(proto::Content {
                        role: String::new(),
                        parts: vec![proto::Part {
                            data: Some(proto::part::Data::Text(doc.clone())),
                            thought: false,
                            thought_signature: Vec::new(),
                        }],
                    }),
                    task_type: None,
                    title: None,
                    output_dimensionality: Some(ndims as i32),
                    auto_truncate: None,
                };

                let mut tonic_req = tonic::Request::new(request);
                client
                    .apply_auth(&mut tonic_req)
                    .await
                    .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?;

                let response = grpc_client
                    .embed_content(tonic_req)
                    .await
                    .map_err(|e| EmbeddingError::ProviderError(e.to_string()))?
                    .into_inner();

                if let Some(ref usage) = response.usage_metadata {
                    total_usage.input_tokens += usage.prompt_token_count as u64;
                    total_usage.output_tokens += usage.candidates_token_count as u64;
                    total_usage.total_tokens += usage.total_token_count as u64;
                    total_usage.cached_input_tokens += usage.cached_content_token_count as u64;
                }

                let Some(embedding) = response.embedding else {
                    return Err(EmbeddingError::ResponseError(
                        "No embedding in response".to_string(),
                    ));
                };

                embeddings_out.push(embeddings::Embedding {
                    document: doc,
                    vec: embedding.values.into_iter().map(|v| v as f64).collect(),
                });
            }

            let span = tracing::Span::current();
            span.record("gen_ai.usage.input_tokens", total_usage.input_tokens);
            span.record("gen_ai.usage.output_tokens", total_usage.output_tokens);

            Ok(embeddings_out)
        };

        async_block.instrument(span).await
    }
}
