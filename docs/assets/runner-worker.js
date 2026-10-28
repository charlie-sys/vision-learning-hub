// Original course runner. Pinned Pyodide API: https://pyodide.org/en/0.27.7/
let runtime;
let busy = false;
let requestId;
const send = (type, data = {}) => self.postMessage({id: requestId, type, ...data});
const INDEX = 'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/';

async function initialize() {
  if (!runtime) {
    send('status', {text: '首次加载 Python 运行环境，请稍候…'});
    importScripts(INDEX + 'pyodide.js');
    runtime = await loadPyodide({indexURL: INDEX});
    runtime.setStdin({error: true});
  }
  return runtime;
}

self.onmessage = async ({data}) => {
  if (busy || data.type !== 'run') return;
  busy = true;
  requestId = data.id;
  let globals;
  let outputLength = 0;
  try {
    const py = await initialize();
    const write = stream => text => {
      if (outputLength > 64000) return;
      outputLength += text.length + 1;
      send('output', {stream, text: outputLength > 64000 ? '[输出超过64KB，后续省略]' : text});
    };
    py.setStdout({batched: write('stdout')});
    py.setStderr({batched: write('stderr')});
    const packageOptions = {messageCallback: () => {}, errorCallback: write('stderr')};
    globals = py.runPython('dict(__name__="__main__")');
    globals.set('__course_source__', data.code);
    // AST inspection also catches "from torch..." without false positives in comments.
    const unsupported = py.runPython(`
import ast
_tree = ast.parse(__course_source__)
_imports = [a.name.split('.')[0] for n in ast.walk(_tree) if isinstance(n, ast.Import) for a in n.names]
_imports += [(n.module or '').split('.')[0] for n in ast.walk(_tree) if isinstance(n, ast.ImportFrom)]
next((x for x in _imports if x in ('torch', 'torchvision', 'tensorflow')), '')
`, {globals});
    if (unsupported) throw new Error(`${unsupported} 请使用本章的 Colab 实验入口；网页数学运行器支持 NumPy、SciPy、SymPy、Matplotlib 等浏览器兼容库。`);
    send('status', {text: '正在准备所需数学库…'});
    // Agg avoids interactive GUI dependencies; figures are sent back as PNGs.
    if (/matplotlib/.test(data.code)) {
      await py.loadPackage('matplotlib', packageOptions);
      py.runPython("import matplotlib; matplotlib.use('Agg', force=True)", {globals});
      py.runPython("import matplotlib.pyplot as _course_plt; _course_plt.show = lambda *args, **kwargs: None", {globals});
    }
    await py.loadPackagesFromImports(data.code, packageOptions);
    send('running', {text: '正在运行…'});
    await py.runPythonAsync(data.code, {globals});
    py.runPython('import sys; sys.stdout.flush(); sys.stderr.flush()', {globals});
    const images = py.runPython(`
import sys as _sys, json as _json, io as _io, base64 as _b64
_images = []
if 'matplotlib.pyplot' in _sys.modules:
    import matplotlib.pyplot as _plt
    for _number in _plt.get_fignums()[:4]:
        _buf = _io.BytesIO()
        _plt.figure(_number).savefig(_buf, format='png', dpi=110, bbox_inches='tight')
        if _buf.tell() < 3 * 1024 * 1024:
            _images.append(_b64.b64encode(_buf.getvalue()).decode('ascii'))
    _plt.close('all')
_json.dumps(_images)
`, {globals});
    send('done', {images: JSON.parse(images)});
  } catch (error) {
    send('error', {text: String(error.message || error)});
  } finally {
    if (runtime) {
      try { runtime.runPython("import sys; sys.stdout.flush(); sys.stderr.flush();\nif 'matplotlib.pyplot' in sys.modules: sys.modules['matplotlib.pyplot'].close('all')"); } catch (_) { /* worker may have failed to load */ }
    }
    if (globals) globals.destroy();
    busy = false;
  }
};
