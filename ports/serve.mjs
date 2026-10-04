// A static server for the ports (the WebGPU page fetches the module, the WGSL and the vectors).
// Usage: node serve.mjs [port], then open http://127.0.0.1:8791/webgpu/test.html. Loopback only.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join, normalize, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('.', import.meta.url));
const port = Number(process.argv[2] ?? process.env.PORT ?? 8791);
const types = { '.html': 'text/html', '.mjs': 'text/javascript', '.js': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.wgsl': 'text/plain' };

createServer(async (request, response) => {
  const path = normalize(join(root, decodeURIComponent(new URL(request.url, 'http://x').pathname)));
  if (!path.startsWith(root) || path.includes(`${sep}node_modules${sep}`)) {
    response.writeHead(403).end();
    return;
  }
  try {
    const body = await readFile(path.endsWith(sep) ? join(path, 'index.html') : path);
    response.writeHead(200, { 'content-type': types[extname(path)] ?? 'application/octet-stream', 'cache-control': 'no-store' }).end(body);
  } catch {
    response.writeHead(404).end();
  }
}).listen(port, '127.0.0.1', () => console.log(`http://127.0.0.1:${port}/webgpu/test.html`));
