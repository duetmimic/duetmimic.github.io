// DUETBIN1 container reader and byte fetching (browser, worker and Node).
// Format: see tools/duet_export/container.py (both sides must stay in step).

const MAGIC = 'DUETBIN1';
const ALIGN = 16;
const TYPES = {
  float32: Float32Array, float64: Float64Array,
  int8: Int8Array, uint8: Uint8Array,
  int16: Int16Array, uint16: Uint16Array,
  int32: Int32Array, uint32: Uint32Array,
};

const isNode = typeof process !== 'undefined' && !!(process.versions && process.versions.node);

function isGzip(bytes) {
  return bytes.length > 2 && bytes[0] === 0x1f && bytes[1] === 0x8b;
}

export async function gunzip(bytes) {
  if (isNode) {
    const zlib = await import('node:zlib');
    const out = zlib.gunzipSync(bytes);
    return new Uint8Array(out.buffer, out.byteOffset, out.byteLength);
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

const message = (error) => (error && error.message) || String(error);

// The error with the URL in front, so a failure says which file broke. An
// AbortError stays as it is, so callers can tell an abort from a failure.
function withURL(href, error) {
  if (error && error.name === 'AbortError') return error;
  return new Error(`${href}: ${message(error)}`, { cause: error });
}

// Fetch a URL (or a file: URL under Node) as bytes, gunzipping when the payload
// is gzip (a server that already applied Content-Encoding hands us plain bytes).
// onProgress(received, total) reports transfer progress when the size is known.
// init goes to fetch() ({signal, cache, ...}); under Node its signal also
// aborts a file read. Every error names the URL: an HTTP status, a network
// failure, or "corrupt or truncated gzip (...)".
export async function fetchBytes(url, onProgress, init) {
  const href = url instanceof URL ? url.href : String(url);
  let bytes;
  if (isNode && href.startsWith('file:')) {
    const fs = await import('node:fs/promises');
    const signal = init && init.signal;
    const buf = await fs.readFile(new URL(href), signal ? { signal } : undefined);
    bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
    if (onProgress) onProgress(bytes.length, bytes.length);
  } else {
    let response;
    try {
      response = await fetch(href, init);
    } catch (error) {
      throw withURL(href, error);
    }
    if (!response.ok) throw new Error(`${href}: HTTP ${response.status}`);
    const total = Number(response.headers.get('content-length')) || 0;
    try {
      if (!response.body || !onProgress) {
        bytes = new Uint8Array(await response.arrayBuffer());
      } else {
        const reader = response.body.getReader();
        const parts = [];
        let received = 0;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          parts.push(value);
          received += value.length;
          onProgress(received, total);
        }
        bytes = new Uint8Array(received);
        let at = 0;
        for (const part of parts) { bytes.set(part, at); at += part.length; }
      }
    } catch (error) {
      throw withURL(href, error); // the connection dropped, or the fetch was aborted
    }
  }
  if (!isGzip(bytes)) return bytes;
  try {
    return await gunzip(bytes);
  } catch (error) {
    throw new Error(`${href}: corrupt or truncated gzip (${message(error)})`, { cause: error });
  }
}

// {meta, arrays: {name: TypedArray}} from uncompressed container bytes. The
// arrays are views into one fresh, aligned ArrayBuffer.
export function parseContainer(input) {
  let bytes = input;
  if (bytes.byteOffset % ALIGN !== 0) bytes = bytes.slice();  // realign
  const magic = new TextDecoder().decode(bytes.subarray(0, 8));
  if (magic !== MAGIC) throw new Error('Not a DUETBIN1 container');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const length = view.getUint32(8, true);
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(12, 12 + length)));
  let start = 12 + length;
  start += (ALIGN - (start % ALIGN)) % ALIGN;
  const arrays = {};
  for (const [name, entry] of Object.entries(header.arrays)) {
    const Type = TYPES[entry.dtype];
    if (!Type) throw new Error(`${name}: unsupported dtype ${entry.dtype}`);
    const begin = bytes.byteOffset + start + entry.offset;
    const count = entry.nbytes / Type.BYTES_PER_ELEMENT;
    const array = new Type(bytes.buffer, begin, count);
    array.shape = entry.shape;
    arrays[name] = array;
  }
  return { meta: header.meta, arrays };
}

// fetchBytes + parseContainer; init as in fetchBytes.
export async function loadContainer(url, onProgress, init) {
  const bytes = await fetchBytes(url, onProgress, init);
  try {
    return parseContainer(bytes);
  } catch (error) {
    throw withURL(url instanceof URL ? url.href : String(url), error);
  }
}

// A JSON file; init as in fetchBytes (demo.json is fetched with {cache: 'no-cache'}).
export async function loadJSON(url, init) {
  const bytes = await fetchBytes(url, undefined, init);
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch (error) {
    throw withURL(url instanceof URL ? url.href : String(url), error);
  }
}
