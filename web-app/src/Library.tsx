import { ImageViewer } from './ImageViewer.tsx'
import { useEffect, useRef, useState } from 'react'
import { ApiError, api, del, patch, post, type Asset, type AssetKind } from './api.ts'
import { AutoTextarea } from './ui.tsx'
import './Library.css'

const kinds: [AssetKind, string, string][] = [
  ['character', 'Characters', 'People and creatures who appear in your films'],
  ['prop', 'Props', 'Objects that must look the same in every shot'],
  ['location', 'Locations', 'Places and sets you return to'],
  ['style', 'Styles', 'Looks you can apply to a whole film'],
  ['music', 'Music', 'Tracks and moods for the Sound Designer'],
]
const singular: Record<AssetKind, string> = { character: 'character', prop: 'prop', location: 'location', style: 'style', music: 'track' }
const icon: Record<AssetKind, string> = { character: '☺', prop: '◆', location: '⌂', style: '✦', music: '♪' }
const working = (asset: Asset) => ['queued', 'running'].includes(asset.generation?.status ?? '') || ['queued', 'training'].includes(asset.lora?.status ?? '')
const pct = (value?: number) => `${Math.round(100 * (value ?? 0))}%`

function LoraBadge({ asset }: { asset: Asset }) {
  const lora = asset.lora
  if (!lora || lora.status === 'none') return null
  if (lora.status === 'ready') return <span className="lib-badge ready">Trained ✓</span>
  if (lora.status === 'failed') return <span className="lib-badge failed">Training failed</span>
  return <span className="lib-badge training">{lora.status === 'queued' ? 'Training queued' : `Training ${pct(lora.progress)}`}</span>
}

export function LibraryView({ onCount }: { onCount: (count: number) => void }) {
  const [kind, setKind] = useState<AssetKind>('character')
  const [assets, setAssets] = useState<Asset[] | null>(null)
  const [missing, setMissing] = useState(false)
  const [openId, setOpenId] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)

  const load = () => api<Asset[]>('/api/library').then((list) => { setAssets(list); setMissing(false); onCount(list.length) })
    .catch((error) => { setAssets([]); setMissing(error instanceof ApiError && error.status === 404) })
  useEffect(() => { load() }, []) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!assets?.some(working)) return
    const timer = setInterval(load, 4000)
    return () => clearInterval(timer)
  }, [assets]) // eslint-disable-line react-hooks/exhaustive-deps

  const shown = (assets ?? []).filter((asset) => asset.kind === kind)
  const count = (k: AssetKind) => (assets ?? []).filter((asset) => asset.kind === k).length
  const meta = kinds.find(([k]) => k === kind)!

  return <div className="library">
    <div className="intro compact"><h1>Characters &amp; Resources<span>.</span></h1></div>
    <p className="subtitle lib-sub">Build a cast and a kit once, then use them in any film. Mention one with <code>@Name</code> when you brief the crew.</p>
    <div className="tabs" role="tablist">{kinds.map(([k, label]) => <button key={k} role="tab" aria-selected={kind === k} className={`tab ${kind === k ? 'active' : ''}`} onClick={() => setKind(k)}>{label} <b>{count(k)}</b></button>)}
      <button className="send-button lib-new" onClick={() => setCreating(true)} disabled={missing}>+ New {singular[kind]}</button>
    </div>
    <p className="tab-hint">{meta[2]}</p>
    {assets === null ? <div className="lib-empty"><i className="file-spinner" /> Loading the library…</div>
      : missing ? <div className="lib-empty"><strong>The library isn't available yet</strong><p>This server doesn't have the Characters &amp; Resources API. Restart <code>rendvid serve</code> after updating.</p></div>
      : shown.length === 0 ? <div className="lib-empty"><span className="spark">✦</span><strong>No {meta[1].toLowerCase()} yet</strong><p>{kind === 'music' ? 'Upload a track or describe a mood.' : `Create a ${singular[kind]}, describe it, then upload or generate reference images.`}</p><button className="send-button" onClick={() => setCreating(true)}>+ New {singular[kind]}</button></div>
      : <div className="lib-grid">{shown.map((asset) => <button key={asset.id} className="lib-card" onClick={() => setOpenId(asset.id)}>
        <div className="lib-cover">{asset.cover ? <img src={asset.cover} alt="" loading="lazy" /> : <span>{icon[asset.kind]}</span>}<LoraBadge asset={asset} />
          {['queued', 'running'].includes(asset.generation?.status ?? '') && <span className="lib-badge gen">Drawing {pct(asset.generation?.progress)}</span>}</div>
        <div className="lib-card-info"><strong>{asset.name}</strong><small>{asset.kind === 'music' ? `${asset.audio?.length ?? 0} tracks` : `${asset.images.length} images`}{asset.used_in?.length ? ` · in ${asset.used_in.length} films` : ''}</small></div>
      </button>)}</div>}
    {creating && <NewAsset kind={kind} onClose={() => setCreating(false)} onCreated={(asset) => { setCreating(false); load(); setOpenId(asset.id) }} />}
    {openId && <AssetDrawer id={openId} onClose={() => { setOpenId(null); load() }} onDeleted={() => { setOpenId(null); load() }} />}
  </div>
}

function NewAsset({ kind, onClose, onCreated }: { kind: AssetKind; onClose: () => void; onCreated: (asset: Asset) => void }) {
  const [name, setName] = useState('')
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const create = async () => {
    if (!name.trim()) return
    setSaving(true)
    try { onCreated(await post<Asset>('/api/library', { kind, name: name.trim() })) } catch (e) { setError((e as Error).message); setSaving(false) }
  }
  return <div className="modal-backdrop" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <div className="modal" role="dialog" aria-modal="true" aria-labelledby="new-title">
      <div className="modal-head"><div><h2 id="new-title">New {singular[kind]}</h2><p>Give it a name you'll mention in briefs, like <code>@Kalu</code> or <code>@Village_Well</code>.</p></div><button className="icon-button" aria-label="Close" onClick={onClose}>✕</button></div>
      <input className="field" autoFocus value={name} onChange={(event) => setName(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') create() }} placeholder="Name" />
      {error && <div className="notice">{error}</div>}
      <div className="modal-actions"><button className="text-button" onClick={onClose}>Cancel</button><button className="send-button" onClick={create} disabled={saving || !name.trim()}>{saving ? 'Creating…' : 'Create'} <span>↗</span></button></div>
    </div>
  </div>
}

function AssetDrawer({ id, onClose, onDeleted }: { id: string; onClose: () => void; onDeleted: () => void }) {
  const [asset, setAsset] = useState<Asset | null>(null)
  const [error, setError] = useState('')
  const [draft, setDraft] = useState({ name: '', description: '', tags: '' })
  const [prompt, setPrompt] = useState('')
  const [describing, setDescribing] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [count, setCount] = useState(8)
  const [view, setView] = useState<number | null>(null)
  const [voices, setVoices] = useState<Record<string, string>>({})
  const fileInput = useRef<HTMLInputElement>(null)

  const take = (next: Asset) => { setAsset(next); setDraft({ name: next.name, description: next.description ?? '', tags: (next.tags ?? []).join(', ') }) }
  const run = async (fn: () => Promise<Asset>) => { setError(''); try { take(await fn()) } catch (e) { setError((e as Error).message) } }
  useEffect(() => { run(() => api<Asset>(`/api/library/${id}`)) }, [id]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { api<Record<string, string>>('/api/library/voices').then(setVoices).catch(() => {}) }, [])
  // Poll while a reference sheet is being drawn or a LoRA is training; keep the user's unsaved edits.
  useEffect(() => {
    if (!asset || !working(asset)) return
    const timer = setInterval(() => api<Asset>(`/api/library/${id}`).then((next) => setAsset(next)).catch(() => {}), 2000)
    return () => clearInterval(timer)
  }, [asset, id])
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  if (!asset) return <div className="drawer-backdrop" onMouseDown={onClose}><aside className="drawer" onMouseDown={(event) => event.stopPropagation()}><div className="lib-empty">{error || <><i className="file-spinner" /> Loading…</>}</div></aside></div>

  const save = (fields: Partial<{ name: string; description: string; tags: string[]; voice: string; cover: string }>) => run(() => patch<Asset>(`/api/library/${id}`, fields))
  const upload = async (files: FileList | File[]) => {
    const list = [...files]
    if (!list.length) return
    const form = new FormData()
    list.forEach((file) => form.append('files', file))
    setUploading(true)
    await run(() => api<Asset>(`/api/library/${id}/images`, { method: 'POST', body: form }))
    setUploading(false)
  }
  const describe = async () => {
    setDescribing(true)
    await run(() => post<Asset>(`/api/library/${id}/describe`, { prompt: prompt.trim() || draft.description }))
    setDescribing(false)
    setPrompt('')
  }
  const remove = async () => {
    if (!confirm(`Delete ${asset.name}? Its images are removed too.`)) return
    try { await del(`/api/library/${id}`); onDeleted() } catch (e) { setError((e as Error).message) }
  }
  const isMusic = asset.kind === 'music'
  const trainable = asset.kind === 'character' || asset.kind === 'style'
  const gen = asset.generation, lora = asset.lora
  const generating = ['queued', 'running'].includes(gen?.status ?? '')
  const training = ['queued', 'training'].includes(lora?.status ?? '')
  const coverFile = asset.cover_file ?? asset.images.find((image) => image.url === asset.cover)?.file
  const viewing = view !== null && asset.images.length ? Math.min(view, asset.images.length - 1) : null

  return <div className="drawer-backdrop" onMouseDown={onClose}>
    <aside className="drawer" role="dialog" aria-modal="true" aria-label={asset.name} onMouseDown={(event) => event.stopPropagation()}>
      <div className="drawer-head">
        <span className="drawer-kind">{icon[asset.kind]} {asset.kind}</span>
        <input className="drawer-name" value={draft.name} onChange={(event) => setDraft({ ...draft, name: event.target.value })} onBlur={() => draft.name.trim() && draft.name !== asset.name && save({ name: draft.name.trim() })} aria-label="Name" />
        <LoraBadge asset={asset} />
        <button className="icon-button" aria-label="Close" onClick={onClose}>✕</button>
      </div>
      <div className="drawer-body">
        {error && <div className="notice">{error}</div>}

        <section className="field-group">
          <label className="field-label">{isMusic ? 'Mood note' : 'Fixed visual description'}</label>
          <p className="field-help">{isMusic ? 'What this music should feel like. The Sound Designer reads it.' : 'In English. Every shot that uses this reads it word for word, so be specific: build, colours, clothes, marks.'}</p>
          <AutoTextarea className="field" rows={4} maxRows={12} value={draft.description} onChange={(event) => setDraft({ ...draft, description: event.target.value })} onBlur={() => draft.description !== asset.description && save({ description: draft.description })} placeholder={isMusic ? 'Gentle bansuri over tabla, warm and unhurried' : 'A small Indian house crow, glossy blue-black feathers, grey neck…'} />
          <div className="ai-row">
            <input className="field" value={prompt} onChange={(event) => setPrompt(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') describe() }} placeholder={`Short idea, e.g. “clever village crow, a bit scruffy”`} disabled={describing} />
            <button className="send-button" onClick={describe} disabled={describing || (!prompt.trim() && !draft.description.trim())}>{describing ? <>Writing<span className="dots inline"><i /><i /><i /></span></> : '✨ Write with AI'}</button>
          </div>
          {describing && <small className="field-help">The Production Designer is writing the description (10–30 s)…</small>}
        </section>

        <section className="field-group two">
          <div><label className="field-label">Tags</label>
            <input className="field" value={draft.tags} onChange={(event) => setDraft({ ...draft, tags: event.target.value })} onBlur={() => draft.tags !== asset.tags.join(', ') && save({ tags: draft.tags.split(',').map((tag) => tag.trim()).filter(Boolean) })} placeholder="hero, bird, village" /></div>
          {asset.kind === 'character' && <div><label className="field-label">Voice</label>
            <select className="field" value={asset.voice ?? ''} onChange={(event) => save({ voice: event.target.value })}>
              <option value="">Narrator decides</option>
              {Object.entries(voices).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select></div>}
        </section>

        <section className="field-group">
          <label className="field-label">{isMusic ? 'Tracks' : 'Reference images'} <b>{isMusic ? asset.audio?.length ?? 0 : asset.images.length}</b></label>
          <div className={`dropzone ${dragging ? 'dragging' : ''}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)}
            onDrop={(event) => { event.preventDefault(); setDragging(false); upload(event.dataTransfer.files) }}>
            <input ref={fileInput} type="file" multiple accept={isMusic ? 'audio/*' : 'image/*'} hidden onChange={(event) => { if (event.target.files) upload(event.target.files); event.target.value = '' }} />
            {uploading ? <span><i className="file-spinner" /> Uploading…</span> : <span>Drop {isMusic ? 'audio files' : 'images'} here or <button className="link-button" onClick={() => fileInput.current?.click()}>choose files</button></span>}
          </div>
          {isMusic ? <div className="tracks">{(asset.audio ?? []).map((track) => <div className="track" key={track.file}><span>{track.file}</span><audio controls preload="none" src={track.url} /><button className="file-remove" title="Delete" onClick={() => run(() => del<Asset>(`/api/library/${id}/images/${encodeURIComponent(track.file)}`))}>✕</button></div>)}</div>
            : <div className="gallery">{asset.images.map((image) => <figure key={image.file} className={coverFile === image.file ? 'is-cover' : ''}>
              <button className="gallery-open" onClick={() => setView(asset.images.indexOf(image))} aria-label="View image"><img src={image.url} alt="" loading="lazy" /></button>
              {image.source === 'generated' && <span className="gen-tag">AI</span>}
              <figcaption>
                {coverFile === image.file ? <span className="cover-tag">Cover</span> : <button onClick={() => save({ cover: image.file })}>Set as cover</button>}
                <button className="danger" aria-label="Delete image" onClick={() => run(() => del<Asset>(`/api/library/${id}/images/${encodeURIComponent(image.file)}`))}>✕</button>
              </figcaption>
            </figure>)}</div>}
        </section>

        {viewing !== null && <ImageViewer
          images={asset.images.map((image) => ({ url: image.url, name: image.file, label: `${image.source === 'generated' ? 'AI generated' : 'Uploaded'}${coverFile === image.file ? ' · Cover' : ''}` }))}
          index={viewing} onIndex={setView} onClose={() => setView(null)}
          actions={(_, i) => {
            const image = asset.images[i]
            if (!image) return null
            return <>
              {coverFile === image.file ? <span className="viewer-tag">★ Cover</span>
                : <button className="viewer-btn primary" onClick={() => save({ cover: image.file })}>★ Set as cover</button>}
              <button className="viewer-btn danger" onClick={async () => {
                if (!confirm('Delete this image?')) return
                await run(() => del<Asset>(`/api/library/${id}/images/${encodeURIComponent(image.file)}`))
                if (asset.images.length <= 1) setView(null)
              }}>🗑 Delete</button>
            </>
          }} />}

        {!isMusic && <section className="field-group job-box">
          <div className="job-head"><div><strong>Generate reference images</strong><p className="field-help">Draws a reference sheet in varied poses and angles from the description. Runs on this Mac's GPU.</p></div>
            <div className="job-controls"><select className="field small" value={count} onChange={(event) => setCount(Number(event.target.value))} disabled={generating}>{[4, 6, 8, 12, 16].map((n) => <option key={n} value={n}>{n} images</option>)}</select>
              <button className="send-button" disabled={generating || !draft.description.trim()} onClick={() => run(() => post<Asset>(`/api/library/${id}/generate`, { count }))}>{generating ? 'Drawing…' : 'Generate'}</button></div></div>
          {gen && gen.status !== 'idle' && <div className={`job-progress ${gen.status}`}><div className="bar"><i style={{ width: pct(gen.progress) }} /></div><small>{gen.status === 'failed' ? `Failed: ${gen.message ?? ''}` : `${gen.message || gen.status} · ${pct(gen.progress)}`}</small></div>}
        </section>}

        {trainable && <section className="field-group job-box">
          <div className="job-head"><div><strong>Train a LoRA</strong><p className="field-help">Teaches the image model this exact {asset.kind} so it looks identical in every shot. Needs at least 6 images; takes about 15–60 min on this Mac.</p></div>
            <button className="send-button" disabled={training || asset.images.length < 6} title={asset.images.length < 6 ? `Add ${6 - asset.images.length} more images first` : ''} onClick={() => run(() => post<Asset>(`/api/library/${id}/train`, {}))}>{training ? 'Training…' : lora?.status === 'ready' ? 'Retrain' : 'Train LoRA'}</button></div>
          {lora && lora.status !== 'none' && <div className={`job-progress ${lora.status}`}><div className="bar"><i style={{ width: lora.status === 'ready' ? '100%' : pct(lora.progress) }} /></div><small>{lora.status === 'ready' ? `Ready${lora.trigger ? ` · trigger word “${lora.trigger}”` : ''}${lora.steps ? ` · ${lora.steps} steps` : ''}` : lora.status === 'failed' ? `Failed: ${lora.message ?? ''}` : `${lora.message || lora.status} · ${pct(lora.progress)}`}</small></div>}
        </section>}

        <div className="drawer-foot"><small>{asset.used_in?.length ? `Used in ${asset.used_in.length} film${asset.used_in.length > 1 ? 's' : ''}` : 'Not used in a film yet'}</small><button className="cancel-button" onClick={remove}>Delete {singular[asset.kind]}</button></div>
      </div>
    </aside>
  </div>
}
