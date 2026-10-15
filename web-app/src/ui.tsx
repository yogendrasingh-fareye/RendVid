import { useEffect, useLayoutEffect, useRef, useState, type ReactNode, type RefObject, type TextareaHTMLAttributes } from 'react'
import Markdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { api, metaOf, type Asset } from './api.ts'

/** A textarea that grows with its content up to maxRows (same behaviour as Cowork's). */
export function AutoTextarea({ maxRows = 6, textareaRef, ...props }: TextareaHTMLAttributes<HTMLTextAreaElement> & { maxRows?: number; textareaRef?: RefObject<HTMLTextAreaElement | null> }) {
  const own = useRef<HTMLTextAreaElement>(null)
  const ref = textareaRef ?? own
  useLayoutEffect(() => {
    const element = ref.current
    if (!element) return
    const style = getComputedStyle(element)
    const line = parseFloat(style.lineHeight) || parseFloat(style.fontSize) * 1.5
    const max = line * maxRows + parseFloat(style.paddingTop) + parseFloat(style.paddingBottom)
    element.style.height = 'auto'
    const wanted = element.scrollHeight
    element.style.height = `${Math.min(wanted, max)}px`
    element.style.overflowY = wanted > max ? 'auto' : 'hidden'
  }, [props.value, maxRows, ref])
  return <textarea ref={ref} {...props} />
}

export function Collapse({ open, children }: { open: boolean; children: ReactNode }) {
  return <div className={`collapse ${open ? 'open' : ''}`} inert={!open}><div className="collapse-inner">{children}</div></div>
}

export function Avatar({ member, small }: { member: string; small?: boolean }) {
  if (member === 'user') return <span className={`user-avatar ${small ? 'small' : ''}`}>Y</span>
  if (member === 'system') return <span className={`agent-avatar slate ${small ? 'small' : ''}`}>R</span>
  const meta = metaOf(member)
  return <span className={`agent-avatar ${meta.color} ${small ? 'small' : ''}`}>{meta.initials}</span>
}

/** Crew messages are chat text: "• " bullets become list items and single newlines stay line breaks. */
const chatMarkdown = (text: string) => text.split('\n').map((line) => line.replace(/^(\s*)• /, '$1- ')).map((line, i, lines) =>
  i < lines.length - 1 && line.trim() && lines[i + 1].trim() && !/^\s*([-*+]|\d+\.) /.test(lines[i + 1]) ? `${line}  ` : line).join('\n')

export function Rich({ text, className = '', chat }: { text: string; className?: string; chat?: boolean }) {
  return <div className={`rich ${className}`}><Markdown remarkPlugins={[remarkGfm]}>{chat ? chatMarkdown(text) : text}</Markdown></div>
}

export function Dots() {
  return <span className="dots" aria-hidden="true"><i /><i /><i /></span>
}

const kindIcon: Record<string, string> = { character: '☺', prop: '◆', location: '⌂', style: '✦', music: '♪' }

/**
 * Textarea with "@" autocomplete over the Characters & Resources library.
 * Typing @ opens a list; ↑/↓ moves, Enter/Tab inserts "@Name". All other keys go to onKeyDown.
 */
export function MentionTextarea({ value, onChange, onKeyDown, ...props }: Omit<TextareaHTMLAttributes<HTMLTextAreaElement>, 'onChange' | 'value'> & { value: string; onChange: (value: string) => void; maxRows?: number }) {
  const ref = useRef<HTMLTextAreaElement>(null)
  const [assets, setAssets] = useState<Asset[] | null>(null)
  const [query, setQuery] = useState<{ text: string; start: number } | null>(null)
  const [index, setIndex] = useState(0)
  useEffect(() => {
    if (query && assets === null) api<Asset[]>('/api/library').then(setAssets).catch(() => setAssets([]))
  }, [query, assets])
  const matches = query && assets ? assets.filter((asset) => asset.name.toLowerCase().includes(query.text.toLowerCase())).slice(0, 7) : []
  const detect = (text: string, caret: number) => {
    const match = /(^|\s)@([^\s@]*)$/.exec(text.slice(0, caret))
    setQuery(match ? { text: match[2], start: caret - match[2].length - 1 } : null)
    setIndex(0)
    if (match && assets !== null && match[2] === '') api<Asset[]>('/api/library').then(setAssets).catch(() => {})
  }
  const pick = (asset: Asset) => {
    if (!query || !ref.current) return
    const caret = ref.current.selectionStart
    const name = asset.name.replace(/\s+/g, '_')
    const next = `${value.slice(0, query.start)}@${name} ${value.slice(caret)}`
    onChange(next)
    setQuery(null)
    const position = query.start + name.length + 2
    requestAnimationFrame(() => { ref.current?.focus(); ref.current?.setSelectionRange(position, position) })
  }
  return <div className="mention-wrap">
    <AutoTextarea {...props} textareaRef={ref} value={value}
      onChange={(event) => { onChange(event.target.value); detect(event.target.value, event.target.selectionStart) }}
      onBlur={() => setTimeout(() => setQuery(null), 150)}
      onKeyDown={(event) => {
        if (query && matches.length) {
          if (event.key === 'ArrowDown') { event.preventDefault(); setIndex((index + 1) % matches.length); return }
          if (event.key === 'ArrowUp') { event.preventDefault(); setIndex((index - 1 + matches.length) % matches.length); return }
          if ((event.key === 'Enter' && !event.metaKey && !event.ctrlKey) || event.key === 'Tab') { event.preventDefault(); pick(matches[index]); return }
        }
        if (query && event.key === 'Escape') { setQuery(null); return }
        onKeyDown?.(event)
      }} />
    {query && <div className="mention-menu" role="listbox">
      {assets === null ? <div className="mention-empty">Loading the library…</div>
        : matches.length === 0 ? <div className="mention-empty">{assets.length ? `Nothing named “${query.text}”` : 'No characters or resources yet. Add them under Characters & Resources.'}</div>
        : matches.map((asset, i) => <button type="button" key={asset.id} role="option" aria-selected={i === index} className={`mention-item ${i === index ? 'active' : ''}`} onMouseDown={(event) => { event.preventDefault(); pick(asset) }} onMouseEnter={() => setIndex(i)}>
          {asset.cover ? <img src={asset.cover} alt="" /> : <span className="mention-icon">{kindIcon[asset.kind]}</span>}
          <span><strong>{asset.name}</strong><small>{asset.kind}{asset.lora?.status === 'ready' ? ' · trained' : ''}</small></span>
        </button>)}
    </div>}
  </div>
}
