import { useEffect, useState, type ReactNode } from 'react'
import './ImageViewer.css'

export type ViewerImage = { url: string; label?: string; name?: string }

type Props = {
  images: ViewerImage[]
  index: number
  onIndex: (index: number) => void
  onClose: () => void
  /** Extra buttons for the current image (e.g. Set as cover, Delete). */
  actions?: (image: ViewerImage, index: number) => ReactNode
}

/** Full-screen lightbox: arrows/keys to flip, click or Z to zoom, Esc to close. */
export function ImageViewer({ images, index, onIndex, onClose, actions }: Props) {
  const [zoom, setZoom] = useState(false)
  const count = images.length
  const image = images[Math.min(index, count - 1)]
  const go = (step: number) => { setZoom(false); onIndex((index + step + count) % count) }

  useEffect(() => {
    // Capture phase + stopImmediatePropagation: Esc closes the viewer, not the drawer behind it.
    const onKey = (event: KeyboardEvent) => {
      const keys: Record<string, () => void> = {
        Escape: () => (zoom ? setZoom(false) : onClose()),
        ArrowLeft: () => go(-1),
        ArrowRight: () => go(1),
        z: () => setZoom((v) => !v),
        Z: () => setZoom((v) => !v),
      }
      const act = keys[event.key]
      if (!act) return
      event.preventDefault()
      event.stopImmediatePropagation()
      act()
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  })

  if (!image) return null
  return <div className="viewer" role="dialog" aria-modal="true" aria-label="Image viewer"
    onMouseDown={(event) => { event.stopPropagation(); if (event.target === event.currentTarget) onClose() }}>
    <div className="viewer-top">
      <span className="viewer-count">{index + 1} / {count}</span>
      {image.label && <span className="viewer-label">{image.label}</span>}
      <span className="viewer-spacer" />
      {actions?.(image, index)}
      <a className="viewer-btn" href={image.url} download={image.name ?? true} title="Download">↓ Download</a>
      <button className="viewer-btn" onClick={() => setZoom((v) => !v)} title="Zoom (Z)">{zoom ? '⤡ Fit' : '⤢ Zoom'}</button>
      <button className="viewer-btn close" onClick={onClose} aria-label="Close (Esc)">✕</button>
    </div>
    <div className={`viewer-stage ${zoom ? 'zoomed' : ''}`} onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <img key={image.url} src={image.url} alt={image.label ?? ''} onClick={() => setZoom((v) => !v)} draggable={false} />
    </div>
    {count > 1 && <>
      <button className="viewer-nav prev" onClick={() => go(-1)} aria-label="Previous (←)">‹</button>
      <button className="viewer-nav next" onClick={() => go(1)} aria-label="Next (→)">›</button>
      <div className="viewer-strip">{images.map((im, i) =>
        <button key={im.url} className={i === index ? 'on' : ''} onClick={() => { setZoom(false); onIndex(i) }} aria-label={`Image ${i + 1}`}>
          <img src={im.url} alt="" loading="lazy" />
        </button>)}</div>
    </>}
  </div>
}
