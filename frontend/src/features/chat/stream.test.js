import { describe, expect, it } from "vitest";

import { consumeSseResponse } from "./stream";

function responseFromChunks(chunks) {
  const encoder = new TextEncoder();
  const body = new ReadableStream({
    start(controller) {
      chunks.forEach((chunk) => controller.enqueue(encoder.encode(chunk)));
      controller.close();
    },
  });
  return new Response(body);
}

async function collect(chunks) {
  const events = [];
  await consumeSseResponse(responseFromChunks(chunks), (event) => events.push(event));
  return events;
}

describe("consumeSseResponse", () => {
  it("parses events split across network chunks", async () => {
    const events = await collect(['data: {"delta":"Ho', 'la"}\n\ndata: {"done":', "true}\n\n"]);
    expect(events).toEqual([{ delta: "Hola" }, { done: true }]);
  });

  it("normalizes CRLF separators and ignores non-data lines", async () => {
    const events = await collect(['event: message\r\ndata: {"answer":"ok"}\r\n\r\n']);
    expect(events).toEqual([{ answer: "ok" }]);
  });

  it("flushes a trailing event without a final blank line", async () => {
    const events = await collect(['data: {"done":true}']);
    expect(events).toEqual([{ done: true }]);
  });

  it("decodes multibyte characters split between chunks", async () => {
    const bytes = new TextEncoder().encode('data: {"delta":"regulación"}\n\n');
    const split = bytes.indexOf(0xc3) + 1;
    const body = new ReadableStream({
      start(controller) {
        controller.enqueue(bytes.slice(0, split));
        controller.enqueue(bytes.slice(split));
        controller.close();
      },
    });
    const events = [];
    await consumeSseResponse(new Response(body), (event) => events.push(event));
    expect(events).toEqual([{ delta: "regulación" }]);
  });
});
