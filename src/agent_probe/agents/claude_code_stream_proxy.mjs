// Streaming pass-through normalizer for the Claude Code CLI.
//
// Unlike opencode_gateway_proxy.mjs — which asks upstream for a non-streaming
// response and re-synthesises the SSE — this proxy keeps the upstream stream
// intact and only rewrites what the CLI rejects. That distinction matters: this
// gateway never returns a large non-streaming completion (a max_tokens=32000
// request hangs past 5 minutes), while the same request streams fine.
//
// Two upstream defects are repaired, both of which otherwise push the CLI onto
// the non-streaming path:
//
//  1. `thinking` blocks open with an empty `signature` and no `signature_delta`
//     ever follows. The CLI rejects that ("Content block is not a text block"),
//     so we substitute a non-empty signature in `content_block_start` and
//     inject a `signature_delta` before the block closes.
//
//  2. Deltas are sometimes addressed to the wrong `index`. Observed: a
//     `tool_use` block opened at index 1 whose `input_json_delta` claims index
//     0, and a `text` block at index 1 whose `text_delta` also claims index 0 —
//     in both cases index 0 is the thinking block. The CLI matches deltas to
//     blocks by index, so it sees a text/JSON delta land on a thinking block
//     and fails with "Content block is not a text block" (or "... not a
//     input_json block"). We re-address each delta to the open block whose type
//     actually matches it.
import { createServer } from "node:http";

const upstreamBaseUrl = process.env.AGENTPROBE_UPSTREAM_BASE_URL;
const port = Number(process.env.AGENTPROBE_PROXY_PORT || "18081");
// Any non-empty value satisfies the CLI; it never round-trips the signature
// back to this gateway in a way the gateway validates.
const PLACEHOLDER_SIGNATURE = "agentprobe_streamed_signature";

// Which block type each delta kind belongs to, used to detect and repair
// misaddressed deltas. signature_delta is handled separately.
const DELTA_TO_BLOCK = {
  text_delta: "text",
  thinking_delta: "thinking",
  input_json_delta: "tool_use",
};

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
    // anthropic-beta is dropped for the same reason as in the OpenCode proxy:
    // this gateway rejects some beta combinations outright.
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

async function readBody(request) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
}

function formatEvent(eventName, payload) {
  const name = eventName ? `event: ${eventName}\n` : "";
  return `${name}data: ${JSON.stringify(payload)}\n\n`;
}

/** Rewrite one SSE block, returning the text to forward downstream. */
function rewriteEvent(eventName, rawData, state) {
  if (rawData === "[DONE]") {
    return formatEvent(eventName, "[DONE]").replace(
      'data: "[DONE]"',
      "data: [DONE]",
    );
  }

  let payload;
  try {
    payload = JSON.parse(rawData);
  } catch {
    // Not JSON we understand — forward untouched rather than dropping it.
    return `${eventName ? `event: ${eventName}\n` : ""}data: ${rawData}\n\n`;
  }

  const type = payload?.type;

  if (type === "content_block_start") {
    const block = payload.content_block || {};
    state.openBlocks.set(payload.index, block.type);
    if (block.type === "thinking") {
      state.thinkingIndexes.add(payload.index);
      state.signatureSent.delete(payload.index);
      if (!block.signature) {
        block.signature = PLACEHOLDER_SIGNATURE;
        payload.content_block = block;
      }
    }
    return formatEvent(eventName, payload);
  }

  if (type === "content_block_delta") {
    const deltaType = payload.delta?.type;
    // If upstream ever starts sending real signatures, honour them: mark the
    // index as handled and pass the delta through, empty values substituted.
    if (deltaType === "signature_delta") {
      state.signatureSent.add(payload.index);
      if (!payload.delta.signature) {
        payload.delta.signature = PLACEHOLDER_SIGNATURE;
      }
      return formatEvent(eventName, payload);
    }

    // Re-address a delta whose index points at a block of the wrong type. Only
    // the most recently opened matching block is considered, which is what the
    // upstream ordering implies.
    const wantedBlock = DELTA_TO_BLOCK[deltaType];
    if (wantedBlock && state.openBlocks.get(payload.index) !== wantedBlock) {
      let target = null;
      for (const [index, blockType] of state.openBlocks) {
        if (blockType === wantedBlock) target = index;
      }
      if (target !== null) payload.index = target;
    }
    return formatEvent(eventName, payload);
  }

  if (type === "content_block_stop") {
    state.openBlocks.delete(payload.index);
    if (state.thinkingIndexes.has(payload.index)) {
      state.thinkingIndexes.delete(payload.index);
      // The CLI wants a signature_delta before the thinking block closes; this
      // gateway never sends one, so inject it here.
      if (!state.signatureSent.has(payload.index)) {
        const injected = formatEvent("content_block_delta", {
          type: "content_block_delta",
          index: payload.index,
          delta: { type: "signature_delta", signature: PLACEHOLDER_SIGNATURE },
        });
        return injected + formatEvent(eventName, payload);
      }
      state.signatureSent.delete(payload.index);
    }
    return formatEvent(eventName, payload);
  }

  return formatEvent(eventName, payload);
}

/** Split a buffer into complete SSE blocks, keeping any partial tail. */
function drainBlocks(buffer) {
  const blocks = [];
  let index;
  while ((index = buffer.indexOf("\n\n")) !== -1) {
    blocks.push(buffer.slice(0, index));
    buffer = buffer.slice(index + 2);
  }
  return { blocks, rest: buffer };
}

function parseBlock(block) {
  let eventName = "";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  return { eventName, rawData: dataLines.join("\n") };
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
    // Deliberately preserve payload.stream: forcing non-streaming is exactly
    // what makes this gateway hang on large max_tokens.
    const streaming = payload.stream === true;

    const upstreamResponse = await fetch(upstreamUrlFor(request.url), {
      method: "POST",
      headers: requestHeadersFor(request.headers),
      body: JSON.stringify(payload),
    });

    if (!streaming || !upstreamResponse.body) {
      const text = await upstreamResponse.text();
      let result;
      try {
        result = JSON.parse(text);
      } catch {
        response.writeHead(upstreamResponse.status, {
          "content-type": upstreamResponse.headers.get("content-type") || "text/plain",
        });
        return response.end(text);
      }
      // Non-streaming replies carry the signature inline; fill it there too.
      for (const block of result.content || []) {
        if (block?.type === "thinking" && !block.signature) {
          block.signature = PLACEHOLDER_SIGNATURE;
        }
      }
      return sendJson(response, upstreamResponse.status, result);
    }

    if (!upstreamResponse.ok) {
      const text = await upstreamResponse.text();
      response.writeHead(upstreamResponse.status, {
        "content-type": upstreamResponse.headers.get("content-type") || "text/plain",
      });
      return response.end(text);
    }

    response.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
    });

    const state = {
      thinkingIndexes: new Set(),
      signatureSent: new Set(),
      openBlocks: new Map(),
    };
    const decoder = new TextDecoder();
    let buffer = "";
    for await (const chunk of upstreamResponse.body) {
      buffer += decoder.decode(chunk, { stream: true });
      const { blocks, rest } = drainBlocks(buffer);
      buffer = rest;
      for (const block of blocks) {
        if (!block.trim()) continue;
        const { eventName, rawData } = parseBlock(block);
        if (!rawData) continue;
        response.write(rewriteEvent(eventName, rawData, state));
      }
    }
    if (buffer.trim()) {
      const { eventName, rawData } = parseBlock(buffer);
      if (rawData) response.write(rewriteEvent(eventName, rawData, state));
    }
    response.end();
  } catch (error) {
    if (response.headersSent) {
      return response.end();
    }
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
