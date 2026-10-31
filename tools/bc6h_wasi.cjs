// Run a WASI identity binary with a read-only-by-convention corpus preopen.
// Usage: node tools/bc6h_wasi.cjs binary.wasm [corpus-directory]
const { WASI } = require('node:wasi');
const fs = require('node:fs');
const path = require('node:path');
const corpus = process.argv[3];
const wasi = new WASI({ version: 'preview1', args: corpus ? ['identity', '/corpus'] : ['identity'],
  preopens: corpus ? { '/corpus': path.resolve(corpus) } : {} });
WebAssembly.compile(fs.readFileSync(process.argv[2])).then(module =>
  WebAssembly.instantiate(module, { wasi_snapshot_preview1: wasi.wasiImport })
).then(instance => wasi.start(instance));
