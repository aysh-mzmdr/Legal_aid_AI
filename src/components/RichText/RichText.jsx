import styles from './RichText.module.css'

// Renders the light Markdown the model writes: paragraphs, "-"/"*" and "1." lists, "#" headings,
// **bold**, *italic*, and passage citations like [1] or [1][3] as clickable chips.
// Builds React elements only (never raw HTML), so model output cannot inject markup.

const INLINE = /(\*\*[^*]+\*\*|\[\d+(?:\s*,\s*\d+)*\]|\*[^*\s][^*]*\*)/g

function renderInline(text, citeProps, keyPrefix) {
  const out = []
  text.split(INLINE).forEach((part, i) => {
    const key = `${keyPrefix}-${i}`
    if (!part) return
    if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
      out.push(<strong key={key}>{part.slice(2, -2)}</strong>)
    } else if (/^\[\d/.test(part)) {
      part
        .slice(1, -1)
        .split(',')
        .map((n) => Number(n.trim()))
        .forEach((n) => {
          const source = citeProps.sources?.find((s) => s.n === n)
          if (!source) return // a number the model made up: drop it
          out.push(
            <button
              key={`${key}-${n}`}
              type="button"
              className={styles.cite}
              onClick={() => citeProps.onCite?.(source)}
              title={source.label}
              aria-label={`Source ${n}: ${source.label}`}
            >
              {n}
            </button>,
          )
        })
    } else if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
      out.push(<em key={key}>{part.slice(1, -1)}</em>)
    } else {
      out.push(part)
    }
  })
  return out
}

function toBlocks(text) {
  const blocks = []
  for (const raw of text.split('\n')) {
    const line = raw.trimEnd()
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/)
    const numbered = line.match(/^\s*\d+[.)]\s+(.*)$/)
    const heading = line.match(/^\s*#{1,6}\s+(.*)$/)
    const last = blocks[blocks.length - 1]
    if (!line.trim()) {
      blocks.push({ type: 'break' })
    } else if (bullet || numbered) {
      const type = bullet ? 'ul' : 'ol'
      const item = (bullet ?? numbered)[1]
      if (last?.type === type) last.items.push(item)
      else blocks.push({ type, items: [item] })
    } else if (heading) {
      blocks.push({ type: 'heading', text: heading[1] })
    } else if (last?.type === 'p') {
      last.text += ` ${line.trim()}`
    } else {
      blocks.push({ type: 'p', text: line.trim() })
    }
  }
  return blocks.filter((b) => b.type !== 'break')
}

export default function RichText({ text, sources, onCite }) {
  const citeProps = { sources, onCite }
  return (
    <div className={styles.rich}>
      {toBlocks(text).map((block, i) => {
        if (block.type === 'ul' || block.type === 'ol') {
          const List = block.type
          return (
            <List key={i} className={styles.list}>
              {block.items.map((item, j) => (
                <li key={j}>{renderInline(item, citeProps, `${i}-${j}`)}</li>
              ))}
            </List>
          )
        }
        if (block.type === 'heading') {
          return (
            <p key={i} className={styles.heading}>
              {renderInline(block.text, citeProps, `${i}`)}
            </p>
          )
        }
        return <p key={i}>{renderInline(block.text, citeProps, `${i}`)}</p>
      })}
    </div>
  )
}
