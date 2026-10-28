/* A reading assistant: launch verified third-party tools on user request. */
document.addEventListener('DOMContentLoaded', () => {
  const dialog = document.querySelector('#translation-dialog');
  if (!dialog) return;
  const title = dialog.querySelector('#translation-paper-title');
  const pdf = dialog.querySelector('#translation-pdf-url');
  const original = dialog.querySelector('#translation-original');
  const copy = dialog.querySelector('#translation-copy');
  const status = dialog.querySelector('#translation-status');
  let trigger;

  document.querySelectorAll('.translation-trigger').forEach(button => {
    button.addEventListener('click', () => {
      trigger = button;
      title.textContent = button.dataset.title;
      pdf.value = button.dataset.pdf;
      original.href = button.dataset.pdf;
      status.textContent = '';
      dialog.showModal();
    });
  });
  dialog.querySelector('#translation-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('close', () => trigger?.focus());
  copy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(pdf.value);
      status.textContent = '论文链接已复制。';
    } catch {
      pdf.focus();
      pdf.select();
      status.textContent = '请复制上方已选中的论文链接。';
    }
  });
});
