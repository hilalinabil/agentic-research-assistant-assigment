// Minimal, safe Markdown renderer for answers: headings, bullet/numbered lists,
// paragraphs, **bold**, *italic*, `code` and [file.md:Lx-Ly] citation badges.
// Builds React elements only (no innerHTML), so model output cannot inject HTML.

const INLINE = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|\[[^\]]*\.(?:md|txt):L\d+[^\]]*\])/g;

function inline(text) {
  return text.split(INLINE).map((part, i) => {
    if (!part) return null;
    if (part.startsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`")) return <code key={i}>{part.slice(1, -1)}</code>;
    if (part.startsWith("[")) return <span key={i} className="cite">{part.slice(1, -1)}</span>;
    if (part.startsWith("*")) return <em key={i}>{part.slice(1, -1)}</em>;
    return part;
  });
}

export default function Markdown({ text }) {
  const blocks = [];
  let list = null;
  const flush = () => {
    if (list) blocks.push(list);
    list = null;
  };

  text.split("\n").forEach((raw, i) => {
    const line = raw.trimEnd();
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    const bullet = line.match(/^\s*[-*]\s+(.*)$/);
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (heading) {
      flush();
      blocks.push(<h3 key={i}>{inline(heading[2])}</h3>);
    } else if (bullet || numbered) {
      const type = bullet ? "ul" : "ol";
      if (!list || list.type !== type) {
        flush();
        list = { type, items: [], key: i };
      }
      list.items.push(<li key={i}>{inline((bullet || numbered)[1])}</li>);
    } else if (!line.trim() || /^-{3,}$/.test(line.trim())) {
      flush();
    } else {
      flush();
      blocks.push(<p key={i}>{inline(line)}</p>);
    }
  });
  flush();

  return blocks.map((b) =>
    b.type === "ul" ? <ul key={b.key}>{b.items}</ul> : b.type === "ol" ? <ol key={b.key}>{b.items}</ol> : b
  );
}
