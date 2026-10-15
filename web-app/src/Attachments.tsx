import { useRef, useState, type DragEvent } from 'react'
import { ImageViewer } from './ImageViewer.tsx'
import { ApiError } from './api.ts'

export type Attachment = { id: string; name: string; url: string; kind: 'audio' | 'image'; duration?: number | null; size?: number }
type Item = { key: string; name: string; status: 'uploading' | 'done' | 'error'; attachment?: Attachment; error?: string }

export const ACCEPT = 'audio/*,.mp3,.wav,.m4a,.ogg,.flac,.aac,image/png,image/jpeg,image/webp'
export const clock = (seconds?: number | null) => {
  if (seconds == null || !isFinite(seconds)) return ''
  const s = Math.round(seconds)
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

/** Files attached to a crew message: each uploads at once to /api/crew/attachments and becomes a chip. */
export function useAttachments() {
  const [items, setItems] = useState<Item[]>([])
  const [dragging, setDragging] = useState(false)
  const counter = useRef(0)

  const add = (files: FileList | File[]) => {
    for (const file of [...files]) {
      const key = `f${++counter.current}`
      setItems((current) => [...current, { key, name: file.name, status: 'uploading' }])
      const form = new FormData()
      form.append('files', file)
      fetch('/api/crew/attachments', { method: 'POST', body: form })
        .then(async (response) => {
          const body = await response.json().catch(() => ({}))
          if (!response.ok) throw new ApiError(typeof body.detail === 'string' ? body.detail : response.statusText, response.status)
          const attachment = (Array.isArray(body) ? body[0] : body) as Attachment
          setItems((current) => current.map((item) => item.key === key ? { ...item, status: 'done', attachment } : item))
        })
        .catch((error: Error) => setItems((current) => current.map((item) => item.key === key ? { ...item, status: 'error', error: error.message } : item)))
    }
  }
  const remove = (key: string) => setItems((current) => current.filter((item) => item.key !== key))
  const clear = () => setItems([])
  const ids = items.flatMap((item) => item.status === 'done' && item.attachment ? [item.attachment.id] : [])
  const uploading = items.some((item) => item.status === 'uploading')
  const dropProps = {
    onDragOver: (event: DragEvent) => { if (event.dataTransfer.types.includes('Files')) { event.preventDefault(); setDragging(true) } },
    onDragLeave: (event: DragEvent) => { if (!event.currentTarget.contains(event.relatedTarget as Node)) setDragging(false) },
    onDrop: (event: DragEvent) => { if (event.dataTransfer.files.length) { event.preventDefault(); add(event.dataTransfer.files) } setDragging(false) },
  }
  return { items, add, remove, clear, ids, uploading, dragging, dropProps }
}

export function AttachButton({ onFiles, disabled }: { onFiles: (files: FileList) => void; disabled?: boolean }) {
  const input = useRef<HTMLInputElement>(null)
  return <>
    <button type="button" className="attach-button" title="Attach a song or images" aria-label="Attach files" disabled={disabled} onClick={() => input.current?.click()}>
      <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d="m21.4 11.6-8.5 8.5a5.5 5.5 0 0 1-7.8-7.8l8.5-8.5a3.7 3.7 0 0 1 5.2 5.2l-8.5 8.5a1.8 1.8 0 0 1-2.6-2.6l7.8-7.8" /></svg>
    </button>
    <input ref={input} type="file" multiple accept={ACCEPT} hidden onChange={(event) => { if (event.target.files?.length) onFiles(event.target.files); event.target.value = '' }} />
  </>
}

export function PendingFiles({ items, onRemove }: { items: Item[]; onRemove: (key: string) => void }) {
  if (!items.length) return null
  return <div className="pending-files">{items.map((item) => {
    const a = item.attachment
    return <span key={item.key} className={`file-chip ${item.status === 'uploading' ? 'uploading' : ''} ${item.status === 'error' ? 'error' : ''}`} title={item.error ?? item.name}>
      {a?.kind === 'image' ? <img src={a.url} alt="" /> : <span className="file-icon">{item.status === 'error' ? '!' : a?.kind === 'audio' || !a ? '♪' : '▤'}</span>}
      <span className="file-name">{item.name}{a?.duration ? <small> · {clock(a.duration)}</small> : null}{item.status === 'error' ? <small> · {item.error}</small> : null}</span>
      {item.status === 'uploading' && <i className="file-spinner" />}
      <button type="button" className="file-remove" aria-label={`Remove ${item.name}`} onClick={() => onRemove(item.key)}>✕</button>
    </span>
  })}</div>
}

/** Attachments sent with a user message: audio as a compact player chip, images as thumbnails. */
export function SentFiles({ files }: { files?: Attachment[] }) {
  const [view, setView] = useState<number | null>(null)
  if (!files?.length) return null
  const images = files.filter((file) => file.kind === 'image')
  return <div className="sent-files">{view !== null && <ImageViewer images={images.map((f) => ({ url: f.url, name: f.name, label: f.name }))}
      index={view} onIndex={setView} onClose={() => setView(null)} />}
    {files.map((file) => file.kind === 'image'
    ? <button key={file.id} className="sent-image" onClick={() => setView(images.indexOf(file))} title={file.name}><img src={file.url} alt={file.name} /></button>
    : <div key={file.id} className="audio-chip"><div className="audio-chip-head"><span className="file-icon">♪</span><strong title={file.name}>{file.name}</strong>{file.duration ? <small>{clock(file.duration)}</small> : null}</div><audio controls preload="none" src={file.url} /></div>)}</div>
}
