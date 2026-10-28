// Validate every generated expression with the same bundled renderer as the site.
const fs = require('node:fs');
const path = require('node:path');
const root = path.resolve(__dirname, '..');
const katex = require(path.join(root, 'vendor/katex/katex.min.js'));
const html = fs.readdirSync(path.join(root, 'docs')).filter(f => f.endsWith('.html'))
  .map(f => fs.readFileSync(path.join(root, 'docs', f), 'utf8')).join('\n');
const decode = s => s.replace(/&quot;/g, '"').replace(/&#x27;/g, "'")
  .replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
const expressions = [...html.matchAll(/data-tex="([^"]*)"/g)];
if (expressions.length < 40) throw new Error('Missing mathematical expressions');
for (const [index, match] of expressions.entries()) {
  try {
    katex.renderToString(decode(match[1]), {throwOnError: true, trust: false});
  } catch (error) {
    throw new Error(`Expression ${index + 1}: ${error.message}`);
  }
}
console.log(`${expressions.length} mathematical expressions render successfully`);
