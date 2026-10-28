(() => {
  'use strict';
  const scriptURL = document.currentScript.src;
  const widgets = [...document.querySelectorAll('.python-runner')];
  if (!widgets.length) return;
  let worker = null;
  let active = null;
  let lastId = 0;
  let timer;
  const state = (widget, text) => { widget.querySelector('.runner-status').textContent = text; };
  const resetControls = () => {
    widgets.forEach(w => {
      w.querySelector('.run-code').disabled = false;
      w.querySelector('.stop-code').disabled = true;
      w.querySelector('.reset-code').disabled = false;
      w.querySelector('textarea').readOnly = false;
      w.removeAttribute('aria-busy');
    });
  };
  function finish() { clearTimeout(timer); active = null; resetControls(); }
  function stop(text = '已停止。运行环境已重置，可再次运行。') {
    if (worker) worker.terminate();
    worker = null;
    if (active) state(active.widget, text);
    finish();
  }
  function deadline(ms, text) { clearTimeout(timer); timer = setTimeout(() => stop(text), ms); }
  function getWorker() {
    if (worker) return worker;
    worker = new Worker(new URL('runner-worker.js', scriptURL));
    worker.onmessage = ({data}) => {
      if (!active || data.id !== active.id) return;
      const widget = active.widget;
      if (data.type === 'status' || data.type === 'running') {
        state(widget, data.text);
        if (data.type === 'running') deadline(30000, '计算超过30秒，已停止。请缩小数据规模或在 Colab 运行。');
      } else if (data.type === 'output') {
        const output = widget.querySelector('.runner-output');
        if (output.textContent.length < 66000) output.textContent += data.text + '\n';
      } else if (data.type === 'error') {
        widget.querySelector('.runner-output').textContent += data.text;
        state(widget, '运行失败；修改代码后可以重试。若运行环境下载失败，请检查网络或使用 Colab。');
        finish();
      } else if (data.type === 'done') {
        const figures = widget.querySelector('.runner-figures');
        for (const [index, png] of data.images.entries()) {
          const image = document.createElement('img');
          image.alt = `Python 运行结果图 ${index + 1}`;
          image.src = 'data:image/png;base64,' + png;
          figures.append(image);
        }
        if (!widget.querySelector('.runner-output').textContent && !data.images.length)
          widget.querySelector('.runner-output').textContent = '运行成功，无打印输出。可使用 print(...) 或 Matplotlib 查看结果。';
        state(widget, `运行完成 · ${((performance.now() - active.started) / 1000).toFixed(1)}秒（含依赖准备）`);
        finish();
      }
    };
    worker.onerror = event => { event.preventDefault(); stop('运行环境无法加载。请检查网络后重试，或打开 Colab 实验。'); };
    worker.onmessageerror = () => stop('运行结果传输失败，环境已重置，请重试。');
    return worker;
  }
  for (const widget of widgets) {
    const editor = widget.querySelector('textarea');
    const original = editor.value;
    const run = () => {
      if (active) return;
      widget.querySelector('.runner-output').textContent = '';
      widget.querySelector('.runner-figures').replaceChildren();
      if (location.protocol === 'file:') {
        state(widget, '请通过在线教程运行，或在项目根目录启动 python -m http.server 8000 --directory docs 后访问 localhost:8000。');
        return;
      }
      if (!editor.value.trim()) { state(widget, '请先输入 Python 代码。'); return; }
      active = {widget, id: ++lastId, started: performance.now()};
      widgets.forEach(w => { w.querySelector('.run-code').disabled = true; });
      widget.querySelector('.stop-code').disabled = false;
      widget.querySelector('.reset-code').disabled = true;
      widget.setAttribute('aria-busy', 'true');
      editor.readOnly = true;
      state(widget, '正在启动 Python…');
      deadline(120000, '环境或数学库加载超过2分钟，已停止。请检查网络后重试，或使用 Colab。');
      try { getWorker().postMessage({type: 'run', id: active.id, code: editor.value}); }
      catch (_) { stop('浏览器未能启动运行环境。请使用支持 Web Worker 的现代浏览器，或打开 Colab。'); }
    };
    widget.querySelector('.run-code').addEventListener('click', run);
    widget.querySelector('.stop-code').addEventListener('click', () => stop());
    widget.querySelector('.reset-code').addEventListener('click', () => {
      editor.value = original;
      widget.querySelector('.runner-output').textContent = '';
      widget.querySelector('.runner-figures').replaceChildren();
      state(widget, '已恢复示例，点击运行查看结果。');
    });
    editor.addEventListener('keydown', event => {
      if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) { event.preventDefault(); run(); }
    });
    widget.querySelector('.download-code').addEventListener('click', () => {
      const blobURL = URL.createObjectURL(new Blob([editor.value], {type: 'text/x-python;charset=utf-8'}));
      const anchor = document.createElement('a');
      anchor.href = blobURL;
      anchor.download = widget.dataset.example + '.py';
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(blobURL), 1000);
    });
  }
})();
