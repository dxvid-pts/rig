// ================================================================
//! Vertex AI gRPC Completion Integration
// ================================================================

use base64::Engine as _;
use rig::OneOrMany;
use rig::completion::{self, CompletionError, CompletionRequest};
use rig::message::{self, MimeType, Reasoning};
use rig::telemetry::ProviderResponseExt;
use rig::telemetry::SpanCombinator;
use std::convert::TryFrom;
use tracing::{Instrument, info_span};

use crate::vertex::client::VertexClient;
use crate::vertex_proto as proto;

// =================================================================
// Rig Implementation Types
// =================================================================

#[derive(Clone, Debug)]
pub struct CompletionModel {
    pub(crate) client: VertexClient,
    pub model: String,
}

impl CompletionModel {
    pub fn new(client: VertexClient, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl completion::CompletionModel for CompletionModel {
    type Response = proto::GenerateContentResponse;
    type StreamingResponse = super::streaming::StreamingCompletionResponse;
    type Client = VertexClient;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<proto::GenerateContentResponse>, CompletionError>
    {
        let request_model = completion_request
            .model
            .clone()
            .unwrap_or_else(|| self.model.clone());

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "generate_content",
                gen_ai.operation.name = "generate_content",
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

        let request = create_grpc_request(&self.client, self.model.clone(), completion_request)?;

        let client = self.client.clone();

        let async_block = async move {
            let mut grpc_client = client.grpc_client();

            let mut tonic_req = tonic::Request::new(request);
            client
                .apply_auth(&mut tonic_req)
                .await
                .map_err(|e| CompletionError::ProviderError(e.to_string()))?;

            let response = grpc_client
                .generate_content(tonic_req)
                .await
                .map_err(|e| CompletionError::ProviderError(e.to_string()))?
                .into_inner();

            let span = tracing::Span::current();
            span.record_response_metadata(&response);
            span.record_token_usage(&response);

            if !response.model_version.is_empty() {
                span.record("gen_ai.response.model", &response.model_version);
            }

            response.try_into()
        };

        async_block.instrument(span).await
    }

    async fn stream(
        &self,
        request: CompletionRequest,
    ) -> Result<rig::streaming::StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>
    {
        super::streaming::stream(self.client.clone(), self.model.clone(), request).await
    }
}

pub(crate) fn create_grpc_request(
    client: &VertexClient,
    default_model: String,
    mut completion_request: CompletionRequest,
) -> Result<proto::GenerateContentRequest, CompletionError> {
    let model = completion_request.model.take().unwrap_or(default_model);

    let mut contents = Vec::new();

    for msg in completion_request.chat_history {
        contents.push(rig_message_to_grpc_content(msg)?);
    }

    let system_instruction = completion_request.preamble.map(|preamble| proto::Content {
        role: "model".to_string(),
        parts: vec![proto::Part {
            data: Some(proto::part::Data::Text(preamble)),
            thought: false,
            thought_signature: Vec::new(),
        }],
    });

    let generation_config = {
        let mut cfg = proto::GenerationConfig::default();
        let mut has_cfg = false;

        if let Some(t) = completion_request.temperature {
            cfg.temperature = Some(t as f32);
            has_cfg = true;
        }

        if let Some(max_tokens) = completion_request.max_tokens {
            cfg.max_output_tokens = Some(max_tokens as i32);
            has_cfg = true;
        }

        if let Some(schema) = completion_request.output_schema {
            cfg.response_mime_type = "application/json".to_string();
            let schema_value = serde_json::to_value(&schema).map_err(|e| {
                CompletionError::RequestError(format!("Invalid output schema: {e}").into())
            })?;
            cfg.response_json_schema = Some(json_to_prost_value(schema_value));
            has_cfg = true;
        }

        if has_cfg { Some(cfg) } else { None }
    };

    let tools = if completion_request.tools.is_empty() {
        vec![]
    } else {
        let function_declarations = completion_request
            .tools
            .into_iter()
            .map(|tool| proto::FunctionDeclaration {
                name: tool.name,
                description: tool.description,
                parameters_json_schema: Some(json_to_prost_value(tool.parameters)),
                ..Default::default()
            })
            .collect();

        vec![proto::Tool {
            function_declarations,
        }]
    };

    let tool_config = completion_request
        .tool_choice
        .map(|choice| proto::ToolConfig {
            function_calling_config: Some(tool_choice_to_vertex(choice)),
        });

    Ok(proto::GenerateContentRequest {
        model: client.resolve_model_path(&model),
        contents,
        system_instruction,
        cached_content: String::new(),
        tools,
        tool_config,
        labels: Default::default(),
        safety_settings: vec![],
        model_armor_config: None,
        generation_config,
    })
}

fn tool_choice_to_vertex(choice: message::ToolChoice) -> proto::FunctionCallingConfig {
    let mut cfg = proto::FunctionCallingConfig::default();

    match choice {
        message::ToolChoice::Auto => {
            cfg.mode = proto::function_calling_config::Mode::Auto as i32;
        }
        message::ToolChoice::Required => {
            cfg.mode = proto::function_calling_config::Mode::Any as i32;
        }
        message::ToolChoice::None => {
            cfg.mode = proto::function_calling_config::Mode::None as i32;
        }
        message::ToolChoice::Specific { function_names } => {
            cfg.mode = proto::function_calling_config::Mode::Any as i32;
            cfg.allowed_function_names = function_names;
        }
    }

    cfg
}

fn rig_message_to_grpc_content(msg: message::Message) -> Result<proto::Content, CompletionError> {
    match msg {
        message::Message::User { content } => {
            let parts = content
                .into_iter()
                .map(rig_user_content_to_grpc_part)
                .collect::<Result<Vec<_>, _>>()?;

            Ok(proto::Content {
                parts,
                role: "user".to_string(),
            })
        }
        message::Message::Assistant { content, .. } => {
            let parts = content
                .into_iter()
                .map(rig_assistant_content_to_grpc_part)
                .collect::<Result<Vec<_>, _>>()?;

            Ok(proto::Content {
                parts,
                role: "model".to_string(),
            })
        }
    }
}

fn rig_user_content_to_grpc_part(
    content: message::UserContent,
) -> Result<proto::Part, CompletionError> {
    match content {
        message::UserContent::Text(message::Text { text }) => Ok(proto::Part {
            data: Some(proto::part::Data::Text(text)),
            thought: false,
            thought_signature: Vec::new(),
        }),
        message::UserContent::ToolResult(result) => {
            let response_text = match &result.content.first() {
                message::ToolResultContent::Text(t) => t.text.clone(),
                message::ToolResultContent::Image(_) => {
                    return Err(CompletionError::RequestError(
                        "Tool result content must be text".into(),
                    ));
                }
            };

            let result_value: serde_json::Value = serde_json::from_str(&response_text)
                .unwrap_or_else(|_| serde_json::json!(response_text));

            let response_struct =
                json_to_prost_struct(serde_json::json!({ "output": result_value }))?;

            Ok(proto::Part {
                data: Some(proto::part::Data::FunctionResponse(
                    proto::FunctionResponse {
                        name: result.id,
                        response: Some(response_struct),
                    },
                )),
                thought: false,
                thought_signature: Vec::new(),
            })
        }
        message::UserContent::Image(img) => {
            let Some(media_type) = img.media_type else {
                return Err(CompletionError::RequestError(
                    "Media type for image is required for Vertex".into(),
                ));
            };

            match media_type {
                message::ImageMediaType::JPEG
                | message::ImageMediaType::PNG
                | message::ImageMediaType::WEBP
                | message::ImageMediaType::HEIC
                | message::ImageMediaType::HEIF => {}
                _ => {
                    return Err(CompletionError::RequestError(
                        format!("Unsupported image media type {media_type:?}").into(),
                    ));
                }
            }

            let mime_type = media_type.to_mime_type().to_string();

            let data = match img.data {
                message::DocumentSourceKind::Url(file_uri) => {
                    return Ok(proto::Part {
                        data: Some(proto::part::Data::FileData(proto::FileData {
                            mime_type,
                            file_uri,
                        })),
                        thought: false,
                        thought_signature: Vec::new(),
                    });
                }
                message::DocumentSourceKind::Raw(bytes) => bytes,
                message::DocumentSourceKind::Base64(data)
                | message::DocumentSourceKind::String(data) => decode_base64_bytes(&data)?,
                message::DocumentSourceKind::Unknown => {
                    return Err(CompletionError::RequestError(
                        "Image content has no body".into(),
                    ));
                }
                _ => {
                    return Err(CompletionError::RequestError(
                        "Unsupported document source kind".into(),
                    ));
                }
            };

            Ok(proto::Part {
                data: Some(proto::part::Data::InlineData(proto::Blob {
                    mime_type,
                    data,
                })),
                thought: false,
                thought_signature: Vec::new(),
            })
        }
        _ => Err(CompletionError::RequestError(
            "Unsupported user content type".into(),
        )),
    }
}

fn rig_assistant_content_to_grpc_part(
    content: message::AssistantContent,
) -> Result<proto::Part, CompletionError> {
    match content {
        message::AssistantContent::Text(message::Text { text }) => Ok(proto::Part {
            data: Some(proto::part::Data::Text(text)),
            thought: false,
            thought_signature: Vec::new(),
        }),
        message::AssistantContent::ToolCall(tool_call) => {
            let args = json_to_prost_struct(tool_call.function.arguments)?;

            Ok(proto::Part {
                data: Some(proto::part::Data::FunctionCall(proto::FunctionCall {
                    name: tool_call.function.name,
                    args: Some(args),
                })),
                thought: false,
                thought_signature: decode_optional_base64(tool_call.signature)?,
            })
        }
        message::AssistantContent::Reasoning(reasoning) => Ok(proto::Part {
            data: Some(proto::part::Data::Text(reasoning.display_text())),
            thought: true,
            thought_signature: decode_optional_base64(
                reasoning.first_signature().map(|s| s.to_string()),
            )?,
        }),
        _ => Err(CompletionError::RequestError(
            "Unsupported assistant content type".into(),
        )),
    }
}

impl TryFrom<proto::GenerateContentResponse>
    for completion::CompletionResponse<proto::GenerateContentResponse>
{
    type Error = CompletionError;

    fn try_from(response: proto::GenerateContentResponse) -> Result<Self, Self::Error> {
        let candidate = response.candidates.first().ok_or_else(|| {
            CompletionError::ResponseError("No response candidates in response".into())
        })?;

        let content_ref = candidate.content.as_ref().ok_or_else(|| {
            CompletionError::ResponseError(format!(
                "Vertex candidate missing content (finish_reason={})",
                candidate.finish_reason
            ))
        })?;

        let mut assistant_contents = Vec::new();

        for part in &content_ref.parts {
            let assistant_content = match &part.data {
                Some(proto::part::Data::Text(text)) => {
                    if part.thought {
                        completion::AssistantContent::Reasoning(Reasoning::new_with_signature(
                            text,
                            encode_optional_base64(&part.thought_signature),
                        ))
                    } else {
                        completion::AssistantContent::text(text)
                    }
                }
                Some(proto::part::Data::InlineData(inline_data)) => {
                    let mime_type = message::MediaType::from_mime_type(&inline_data.mime_type);
                    match mime_type {
                        Some(message::MediaType::Image(media_type)) => {
                            let b64 =
                                base64::engine::general_purpose::STANDARD.encode(&inline_data.data);
                            completion::AssistantContent::image_base64(
                                b64,
                                Some(media_type),
                                Some(message::ImageDetail::default()),
                            )
                        }
                        _ => {
                            return Err(CompletionError::ResponseError(format!(
                                "Unsupported media type {mime_type:?}"
                            )));
                        }
                    }
                }
                Some(proto::part::Data::FunctionCall(function_call)) => {
                    let args = function_call
                        .args
                        .as_ref()
                        .map(prost_struct_to_json)
                        .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));

                    let tool_call = message::ToolCall::new(
                        function_call.name.clone(),
                        message::ToolFunction::new(function_call.name.clone(), args),
                    )
                    .with_signature(encode_optional_base64(&part.thought_signature));

                    completion::AssistantContent::ToolCall(tool_call)
                }
                _ => {
                    return Err(CompletionError::ResponseError(
                        "Response did not contain a message or tool call".into(),
                    ));
                }
            };

            assistant_contents.push(assistant_content);
        }

        let choice = OneOrMany::many(assistant_contents).map_err(|_| {
            CompletionError::ResponseError(
                "Response contained no message or tool call (empty)".to_owned(),
            )
        })?;

        let usage = response
            .usage_metadata
            .as_ref()
            .map(|usage| completion::Usage {
                input_tokens: usage.prompt_token_count as u64,
                output_tokens: usage.candidates_token_count as u64,
                total_tokens: usage.total_token_count as u64,
                cached_input_tokens: usage.cached_content_token_count as u64,
            })
            .unwrap_or_default();

        Ok(completion::CompletionResponse {
            choice,
            usage,
            raw_response: response,
            message_id: None,
        })
    }
}

impl ProviderResponseExt for proto::GenerateContentResponse {
    type OutputMessage = proto::Candidate;
    type Usage = proto::UsageMetadata;

    fn get_response_id(&self) -> Option<String> {
        if self.response_id.is_empty() {
            None
        } else {
            Some(self.response_id.clone())
        }
    }

    fn get_response_model_name(&self) -> Option<String> {
        if self.model_version.is_empty() {
            None
        } else {
            Some(self.model_version.clone())
        }
    }

    fn get_output_messages(&self) -> Vec<Self::OutputMessage> {
        self.candidates.clone()
    }

    fn get_text_response(&self) -> Option<String> {
        self.candidates.first().and_then(|c| {
            c.content.as_ref().and_then(|content| {
                let text: Vec<String> = content
                    .parts
                    .iter()
                    .filter_map(|part| {
                        if let Some(proto::part::Data::Text(text)) = &part.data {
                            Some(text.clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                if text.is_empty() {
                    None
                } else {
                    Some(text.join("\n"))
                }
            })
        })
    }

    fn get_usage(&self) -> Option<Self::Usage> {
        self.usage_metadata
    }
}

fn decode_base64_bytes(input: &str) -> Result<Vec<u8>, CompletionError> {
    let data = input.trim();

    let data = if let Some(rest) = data.strip_prefix("data:") {
        rest.split_once(',').map(|(_, b64)| b64).unwrap_or(data)
    } else {
        data
    };

    let mut last_err: Option<String> = None;

    for engine in [
        &base64::engine::general_purpose::STANDARD,
        &base64::engine::general_purpose::URL_SAFE,
        &base64::engine::general_purpose::STANDARD_NO_PAD,
        &base64::engine::general_purpose::URL_SAFE_NO_PAD,
    ] {
        match engine.decode(data) {
            Ok(bytes) => return Ok(bytes),
            Err(err) => last_err = Some(err.to_string()),
        }
    }

    let err = last_err.unwrap_or_else(|| "unknown base64 decode error".to_string());
    Err(CompletionError::RequestError(
        format!("Invalid base64 data: {err}").into(),
    ))
}

fn decode_optional_base64(sig: Option<String>) -> Result<Vec<u8>, CompletionError> {
    let Some(sig) = sig else {
        return Ok(Vec::new());
    };
    decode_base64_bytes(&sig)
}

fn encode_optional_base64(bytes: &[u8]) -> Option<String> {
    if bytes.is_empty() {
        None
    } else {
        Some(base64::engine::general_purpose::STANDARD.encode(bytes))
    }
}

fn json_to_prost_struct(value: serde_json::Value) -> Result<proto::Struct, CompletionError> {
    match value {
        serde_json::Value::Object(map) => Ok(proto::Struct {
            fields: map
                .into_iter()
                .map(|(k, v)| (k, json_to_prost_value(v)))
                .collect(),
        }),
        _ => Err(CompletionError::RequestError(
            "Expected a JSON object for google.protobuf.Struct".into(),
        )),
    }
}

fn json_to_prost_value(value: serde_json::Value) -> proto::Value {
    match value {
        serde_json::Value::Null => proto::Value {
            kind: Some(proto::value::Kind::NullValue(
                proto::NullValue::NullValue as i32,
            )),
        },
        serde_json::Value::Bool(b) => proto::Value {
            kind: Some(proto::value::Kind::BoolValue(b)),
        },
        serde_json::Value::Number(n) => proto::Value {
            kind: Some(proto::value::Kind::NumberValue(
                n.as_f64().unwrap_or_default(),
            )),
        },
        serde_json::Value::String(s) => proto::Value {
            kind: Some(proto::value::Kind::StringValue(s)),
        },
        serde_json::Value::Array(items) => proto::Value {
            kind: Some(proto::value::Kind::ListValue(proto::ListValue {
                values: items.into_iter().map(json_to_prost_value).collect(),
            })),
        },
        serde_json::Value::Object(map) => proto::Value {
            kind: Some(proto::value::Kind::StructValue(proto::Struct {
                fields: map
                    .into_iter()
                    .map(|(k, v)| (k, json_to_prost_value(v)))
                    .collect(),
            })),
        },
    }
}

fn prost_struct_to_json(st: &proto::Struct) -> serde_json::Value {
    let mut out = serde_json::Map::with_capacity(st.fields.len());
    for (k, v) in &st.fields {
        out.insert(k.clone(), prost_value_to_json(v));
    }
    serde_json::Value::Object(out)
}

fn prost_value_to_json(v: &proto::Value) -> serde_json::Value {
    match &v.kind {
        None | Some(proto::value::Kind::NullValue(_)) => serde_json::Value::Null,
        Some(proto::value::Kind::BoolValue(b)) => serde_json::Value::Bool(*b),
        Some(proto::value::Kind::NumberValue(n)) => serde_json::Number::from_f64(*n)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null),
        Some(proto::value::Kind::StringValue(s)) => serde_json::Value::String(s.clone()),
        Some(proto::value::Kind::StructValue(st)) => prost_struct_to_json(st),
        Some(proto::value::Kind::ListValue(list)) => {
            serde_json::Value::Array(list.values.iter().map(prost_value_to_json).collect())
        }
    }
}
