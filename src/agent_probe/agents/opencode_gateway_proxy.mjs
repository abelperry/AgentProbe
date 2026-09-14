import { createServer } from "node:http";

const upstreamBaseUrl = process.env.AGENTPROBE_UPSTREAM_BASE_URL;
const port = Number(process.env.AGENTPROBE_PROXY_PORT || "18080");

if (!upstreamBaseUrl) {
  throw new Error("AGENTPROBE_UPSTREAM_BASE_URL is required");
}

const hopByHopHeaders = new Set([
  "connection",
  "content-length",
  "host",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

function upstreamUrlFor(requestUrl) {
  const incoming = new URL(requestUrl, "http://127.0.0.1");
  const target = new URL(upstreamBaseUrl);
  const incomingPath = incoming.pathname.replace(/^\/v1(?=\/|$)/, "");
  target.pathname = `${target.pathname.replace(/\/$/, "")}${incomingPath}`;
  target.search = incoming.search;
  return target;
}

function requestHeadersFor(headers) {
  const result = new Headers();
  for (const [name, value] of Object.entries(headers)) {
    const normalized = name.toLowerCase();
    if (hopByHopHeaders.has(normalized) || normalized === "anthropic-beta") {
      continue;
    }
    if (Array.isArray(value)) {
      for (const item of value) result.append(name, item);
    } else if (value !== undefined) {
      result.set(name, value);
    }
  }
  result.set("content-type", "application/json");
  return result;
}

function sendJson(response, status, payload) {
  const body = JSON.stringify(payload);
  response.writeHead(status, {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
  });
  response.end(body);
}

function writeOpenAiSse(response, payload) {
  const choice = payload?.choices?.[0] || {};
  const message = choice.message || {};
  const toolCalls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  const finishReason = choice.finish_reason ?? (toolCalls.length ? "tool_calls" : "stop");
  const id = payload.id || `chatcmpl-agentprobe-${Date.now()}`;
  const created = payload.created || Math.floor(Date.now() / 1000);
  const model = payload.model || "unknown";

  const writeChunk = (delta, reason = null) => {
    response.write(`data: ${JSON.stringify({
      id,
      object: "chat.completion.chunk",
      created,
      model,
      choices: [{ index: 0, delta, finish_reason: reason }],
    })}\n\n`);
  };

  if (toolCalls.length) {
    writeChunk({
      role: "assistant",
      tool_calls: toolCalls.map((toolCall, index) => ({
        index,
        id: toolCall.id || `call_${index}`,
        type: "function",
        function: {
          name: toolCall.function?.name || "",
          arguments: "",
        },
      })),
    });
    for (const [index, toolCall] of toolCalls.entries()) {
      const rawArguments = toolCall.function?.arguments;
      const args = typeof rawArguments === "string"
        ? rawArguments
        : rawArguments == null
          ? ""
          : JSON.stringify(rawArguments);
      if (args) {
        writeChunk({ tool_calls: [{ index, function: { arguments: args } }] });
      }
    }
  } else {
    writeChunk({ role: "assistant", content: "" });
    if (message.content) writeChunk({ content: message.content });
  }
  writeChunk({}, finishReason);
  response.end("data: [DONE]\n\n");
}

function writeAnthropicEvent(response, event, data) {
  response.write(`event: ${event}\r\ndata: ${JSON.stringify(data)}\r\n\r\n`);
}

function writeAnthropicSse(response, payload) {
  const content = Array.isArray(payload.content)
    ? payload.content
    : payload.content
      ? [{ type: "text", text: String(payload.content) }]
      : [];
  const usage = payload.usage || { input_tokens: 0, output_tokens: 0 };

  writeAnthropicEvent(response, "message_start", {
    type: "message_start",
    message: {
      id: payload.id || `msg_agentprobe_${Date.now()}`,
      type: "message",
      role: payload.role || "assistant",
      content: [],
      model: payload.model || "unknown",
      stop_reason: null,
      stop_sequence: payload.stop_sequence ?? null,
      usage,
    },
  });

  content.forEach((block, index) => {
    let startBlock;
    let delta;
    if (block.type === "tool_use") {
      startBlock = {
        type: "tool_use",
        id: block.id || `toolu_agentprobe_${index}`,
        name: block.name || "",
        input: {},
      };
      delta = {
        type: "input_json_delta",
        partial_json: JSON.stringify(block.input || {}),
      };
    } else if (block.type === "thinking") {
      startBlock = { type: "thinking", thinking: "" };
      delta = { type: "thinking_delta", thinking: block.thinking || "" };
    } else if (block.type === "code") {
      startBlock = { type: "code", code: "" };
      delta = { type: "code_delta", code: block.code || "" };
    } else {
      startBlock = { type: "text", text: "" };
      delta = { type: "text_delta", text: block.text || "" };
    }

    writeAnthropicEvent(response, "content_block_start", {
      type: "content_block_start",
      index,
      content_block: startBlock,
    });
    writeAnthropicEvent(response, "content_block_delta", {
      type: "content_block_delta",
      index,
      delta,
    });
    if (block.type === "thinking") {
      writeAnthropicEvent(response, "content_block_delta", {
        type: "content_block_delta",
        index,
        delta: {
          type: "signature_delta",
          signature: block.signature || "dummyinstream",
        },
      });
    }
    writeAnthropicEvent(response, "content_block_stop", {
      type: "content_block_stop",
      index,
    });
  });

  writeAnthropicEvent(response, "message_delta", {
    type: "message_delta",
    delta: {
      stop_reason: payload.stop_reason || "end_turn",
      stop_sequence: payload.stop_sequence ?? null,
    },
    usage,
  });
  writeAnthropicEvent(response, "message_stop", { type: "message_stop" });
  response.end();
}

async function readBody(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
}

const server = createServer(async (request, response) => {
  if (request.url === "/healthz") {
    return sendJson(response, 200, { status: "ok" });
  }
  if (request.method !== "POST") {
    return sendJson(response, 405, { error: { message: "method not allowed" } });
  }

  try {
    const rawBody = await readBody(request);
    const payload = JSON.parse(rawBody || "{}");
    const requestedStreaming = payload.stream === true;
    payload.stream = false;
    const upstreamResponse = await fetch(upstreamUrlFor(request.url), {
      method: "POST",
      headers: requestHeadersFor(request.headers),
      body: JSON.stringify(payload),
    });
    const upstreamBody = await upstreamResponse.text();
    let result;
    try {
      result = JSON.parse(upstreamBody);
    } catch {
      response.writeHead(upstreamResponse.status, {
        "content-type": upstreamResponse.headers.get("content-type") || "text/plain",
      });
      return response.end(upstreamBody);
    }

    if (!upstreamResponse.ok || !requestedStreaming) {
      return sendJson(response, upstreamResponse.status, result);
    }

    response.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
    });
    if (Array.isArray(result.choices)) {
      writeOpenAiSse(response, result);
    } else {
      writeAnthropicSse(response, result);
    }
  } catch (error) {
    sendJson(response, 502, {
      error: {
        type: "gateway_normalization_error",
        message: error instanceof Error ? error.message : String(error),
      },
    });
  }
});

server.listen(port, "127.0.0.1");

function shutdown() {
  server.close(() => process.exit(0));
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
