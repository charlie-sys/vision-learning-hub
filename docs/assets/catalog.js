/* Search and filters share URL state so topic links can be bookmarked. */
(() => {
  const q = document.querySelector('#q');
  const group = document.querySelector('#g');
  const level = document.querySelector('#l');
  const topic = document.querySelector('#s');
  const cards = [...document.querySelectorAll('.grid article')];
  const params = new URLSearchParams(location.search);
  const HALLUCINATION = '06_视觉幻觉';
  q.value = params.get('q') || '';
  for (const [element, name] of [[group,'group'],[level,'level'],[topic,'topic']]) {
    const value = params.get(name) || '';
    element.value = [...element.options].some(option => option.value === value) ? value : '';
  }
  if (topic.value) group.value = HALLUCINATION;
  function filter(event) {
    if (event?.target === topic && topic.value) group.value = HALLUCINATION;
    topic.disabled = Boolean(group.value && group.value !== HALLUCINATION);
    if (topic.disabled) topic.value = '';
    const search = q.value.normalize('NFKC').toLowerCase().trim();
    let visible = 0;
    for (const card of cards) {
      card.hidden = !(card.dataset.search.includes(search) &&
        (!group.value || card.dataset.group === group.value) &&
        (!level.value || card.dataset.level === level.value) &&
        (!topic.value || card.dataset.subcategory === topic.value));
      if (!card.hidden) visible++;
    }
    document.querySelector('#count').textContent = `显示 ${visible} / ${cards.length} 篇`;
    document.querySelector('#empty-results').hidden = visible > 0;
    if (location.protocol !== 'file:') {
      const url = new URL(location.href);
      for (const [name,value] of [['q',q.value],['group',group.value],['level',level.value],['topic',topic.value]]) {
        if (value) url.searchParams.set(name,value); else url.searchParams.delete(name);
      }
      history.replaceState(null,'',url);
    }
  }
  [q,group,level,topic].forEach(element => element.addEventListener('input',filter));
  filter();
})();
