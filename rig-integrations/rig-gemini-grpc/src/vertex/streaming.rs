// ================================================================
//! Vertex AI gRPC Streaming Integration
// ================================================================

use async_stream::stream;
use base64::Engine as _;
use futures::StreamExt;
use serde_json::{Map, Value};

use rig::completion::{CompletionError, CompletionRequest};
use rig::streaming;
use rig::telemetry::SpanCombinator;
use tracing::info_span;
use tracing_futures::Instrument;

use crate::vertex::client::VertexClient;
use crate::vertex_proto as proto;

pub type StreamingCompletionResponse = proto::GenerateContentResponse;

pub(crate) async fn stream(
    client: VertexClient,
    model: String,
    completion_request: CompletionRequest,
) -> Result<streaming::StreamingCompletionResponse<StreamingCompletionResponse>, CompletionError> {
    let request_model = completion_request
        .model
        .clone()
        .unwrap_or_else(|| model.clone());

    let span = if tracing::Span::current().is_disabled() {
        info_span!(
            target: "rig::completions",
            "stream_generate_content",
            gen_ai.operation.name = "stream_generate_content",
            gen_ai.provider.name = "gcp.vertexai",
            gen_ai.request.model = &request_model,
            gen_ai.system_instructions = &completion_request.preamble,
            gen_ai.response.id = tracing::field::Empty,
            gen_ai.response.model = tracing::field::Empty,
            gen_ai.response.model_name = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
            gen_ai.usage.input_tokens = tracing::field::Empty,
        )
    } else {
        tracing::Span::current()
    };

    let request = super::completion::create_grpc_request(&client, model, completion_request)?;

    let mut grpc_client = client.grpc_client();

    let mut tonic_req = tonic::Request::new(request);
    client
        .apply_auth(&mut tonic_req)
        .await
        .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

    let mut response_stream = grpc_client
        .stream_generate_content(tonic_req)
        .await
        .map_err(|e| CompletionError::ProviderError(e.to_string()))?
        .into_inner();

    let stream = stream! {
        let span = tracing::Span::current();
        let mut last_resp: Option<StreamingCompletionResponse> = None;
        let mut final_resp: Option<StreamingCompletionResponse> = None;

        while let Some(item) = response_stream.next().await {
            match item {
                Ok(resp) => {
                    let mut is_final = false;

                    if let Some(candidate) = resp.candidates.first() {
                        // Enum default is 0 = FINISH_REASON_UNSPECIFIED.
                        if candidate.finish_reason != 0 {
                            is_final = true;
                        }

                        if let Some(content) = candidate.content.as_ref() {
                            for part in &content.parts {
                                match &part.data {
                                    Some(proto::part::Data::Text(text)) => {
                                        if part.thought {
                                            yield Ok(streaming::RawStreamingChoice::ReasoningDelta {
                                                id: None,
                                                reasoning: text.clone(),
                                            });
                                        } else {
                                            yield Ok(streaming::RawStreamingChoice::Message(text.clone()));
                                        }
                                    }
                                    Some(proto::part::Data::FunctionCall(function_call)) => {
                                        let args_json = function_call
                                            .args
                                            .as_ref()
                                            .map(prost_struct_to_json)
                                            .unwrap_or_else(|| Value::Object(Map::new()));

                                        let tool_call = streaming::RawStreamingToolCall::new(
                                            function_call.name.clone(),
                                            function_call.name.clone(),
                                            args_json,
                                        )
                                        .with_signature(encode_signature(&part.thought_signature));

                                        yield Ok(streaming::RawStreamingChoice::ToolCall(tool_call));
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }

                    if is_final {
                        final_resp = Some(resp);
                        break;
                    } else {
                        last_resp = Some(resp);
                    }
                }
                Err(status) => {
                    yield Err(CompletionError::ProviderError(status.to_string()));
                    return;
                }
            }
        }

        let resp = final_resp.or(last_resp).unwrap_or_default();
        span.record_response_metadata(&resp);
        span.record_token_usage(&resp);

        if !resp.model_version.is_empty() {
            span.record("gen_ai.response.model", &resp.model_version);
        }
        yield Ok(streaming::RawStreamingChoice::FinalResponse(resp));
    };

    Ok(streaming::StreamingCompletionResponse::stream(Box::pin(
        stream.instrument(span),
    )))
}

fn encode_signature(bytes: &[u8]) -> Option<String> {
    if bytes.is_empty() {
        None
    } else {
        Some(base64::engine::general_purpose::STANDARD.encode(bytes))
    }
}

fn prost_struct_to_json(st: &proto::Struct) -> Value {
    let mut out = Map::with_capacity(st.fields.len());
    for (k, v) in &st.fields {
        out.insert(k.clone(), prost_value_to_json(v));
    }
    Value::Object(out)
}

fn prost_value_to_json(v: &proto::Value) -> Value {
    match &v.kind {
        None | Some(proto::value::Kind::NullValue(_)) => Value::Null,
        Some(proto::value::Kind::BoolValue(b)) => Value::Bool(*b),
        Some(proto::value::Kind::NumberValue(n)) => serde_json::Number::from_f64(*n)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Some(proto::value::Kind::StringValue(s)) => Value::String(s.clone()),
        Some(proto::value::Kind::StructValue(st)) => prost_struct_to_json(st),
        Some(proto::value::Kind::ListValue(list)) => {
            Value::Array(list.values.iter().map(prost_value_to_json).collect())
        }
    }
}
