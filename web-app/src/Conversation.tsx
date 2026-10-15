import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { ago, money, pillOf, post, stageLabel, type Message, type Production, type Roster } from './api.ts'
import { Avatar, Dots, MentionTextarea, Rich } from './ui.tsx'
import { AttachButton, PendingFiles, SentFiles, clock, useAttachments } from './Attachments.tsx'

type Props = { prod: Production; roster: Roster; onChanged: () => void; onError: (message: string) => void }

const live = (prod: Production) => !!prod.job && ['running', 'queued'].includes(prod.job.status)

export function Conversation({ prod, roster, onChanged, onError }: Props) {
  const scroller = useRef<HTMLDivElement>(null)
  const nearBottom = useRef(true)
  const userInput = useRef(0) // last wheel/touch/key/pointer time: only the reader's own scrolling unpins the thread
  const pin = (element: HTMLElement) => { element.scrollTop = element.scrollHeight }
  const touched = () => { userInput.current = Date.now() }
  const [reply, setReply] = useState('')
  const [sending, setSending] = useState(false)
  const files = useAttachments()

  const lastOf = (kind: Message['kind']) => prod.messages.findLast((m) => m.kind === kind)?.id
  const last = { brief: lastOf('brief'), storyboard: lastOf('storyboard'), progress: lastOf('progress'), report: lastOf('report'), stopped: lastOf('stopped') }
  const rendering = live(prod)
  const progressCard = !!last.progress && !!prod.job && (prod.stage === 'rendering' || rendering)
  const activity = prod.busy && prod.activity ? prod.activity : rendering && !progressCard ? `Rendering · ${prod.job?.message || prod.job?.status}` : ''

  // Follow new messages only when the reader is already at the bottom.
  useLayoutEffect(() => {
    const element = scroller.current
    if (element && nearBottom.current) pin(element)
  }, [prod.messages.length, prod.job?.progress, activity])
  // Videos and images grow the thread after they load; stay pinned to the bottom if the reader was there.
  useEffect(() => {
    const element = scroller.current
    const content = element?.firstElementChild
    if (!element || !content) return
    const observer = new ResizeObserver(() => { if (nearBottom.current) pin(element) })
    observer.observe(content)
    return () => observer.disconnect()
  }, [])

  const act = async (fn: () => Promise<unknown>) => {
    try { await fn(); nearBottom.current = true; onChanged() } catch (error) { onError((error as Error).message) }
  }
  const approve = (what: 'brief' | 'storyboard') => act(() => post(`/api/crew/productions/${prod.id}/approve`, { what }))
  const stop = () => act(() => post(`/api/crew/productions/${prod.id}/stop`, {}))
  const resume = () => act(() => post(`/api/crew/productions/${prod.id}/resume`, {}))
  const say = async (text: string, withFiles = true) => {
    text = text.trim()
    const attachments = withFiles ? files.ids : []
    if ((!text && !attachments.length) || sending || (withFiles && files.uploading)) return
    setSending(true)
    await act(async () => {
      await post(`/api/crew/productions/${prod.id}/message`, { text, attachments })
      if (withFiles) { setReply(''); files.clear() }
    })
    setSending(false)
  }

  const who = (m: Message) => m.from === 'user' ? 'You' : m.from === 'system' ? 'RendVid' : roster[m.from]?.role ?? m.from

  function card(m: Message): ReactNode {
    const d = m.data ?? {}
    if (m.kind === 'brief') {
      const b = d.brief ?? {}
      const rows: [string, unknown][] = [['Title', b.title], ['Format', b.format], ['Language', b.language], ['Source', b.source],
        ['Song', b.song && (prod.messages.flatMap((x) => x.data?.attachments ?? []).find((a: { id: string }) => a.id === b.song)?.name ?? b.song)], ['Length', b.minutes && `${b.minutes} min`], ['Aspect', b.aspect], ['Quality', b.quality], ['Style', b.style],
        ['Audience', b.audience], ['Notes', b.notes], ['Idea', b.idea], ['Script', b.script && (b.script.length > 300 ? `${b.script.slice(0, 300)}…` : b.script)]]
      return <div className="card">
        <div className="card-title">Brief</div>
        <table className="kv"><tbody>{rows.filter(([, v]) => v).map(([k, v]) => <tr key={k}><th>{k}</th><td>{String(v)}</td></tr>)}</tbody></table>
        {m.id === last.brief && prod.stage === 'awaiting_brief' && <div className="card-actions"><Button primary onClick={() => approve('brief')}>Approve brief <span>✓</span></Button><span className="card-hint">or reply with changes</span></div>}
      </div>
    }
    if (m.kind === 'storyboard') {
      const rows: { line: number; narration?: string; beat?: string; visual?: string; characters?: string[] }[] = d.rows ?? []
      return <div className="card">
        <div className="card-title">Storyboard <b>{rows.length} lines</b></div>
        <ol className="board">{rows.map((r) => <li key={r.line}>
          <span className="board-n">{r.line}</span>
          <div><p>{r.narration || (r.beat && r.beat !== '(instrumental)' ? <>♪ {r.beat}</> : <em>{r.beat === '(instrumental)' ? '♪ instrumental' : `(silent) ${r.beat ?? ''}`}</em>)}</p>
            <small>🎥 {r.visual}{r.characters?.length ? <> · <span className="chars">👤 {r.characters.join(', ')}</span></> : null}</small></div>
        </li>)}</ol>
        {m.id === last.storyboard && prod.stage === 'awaiting_storyboard' && <div className="card-actions"><Button primary onClick={() => approve('storyboard')}>Approve &amp; render <span>▶</span></Button><span className="card-hint">or reply with changes, e.g. “line 3 should be a close-up”</span></div>}
      </div>
    }
    if (m.kind === 'lyrics') {
      const lines: [number, number, string][] = d.lines ?? []
      return <details className="card script-card"><summary>Lyrics <b>{lines.length ? `${lines.length} lines` : 'instrumental'} · {clock(d.duration)}</b></summary>
        <ol className="board">{lines.map(([start, , text], i) => <li key={i}><span className="board-n">{clock(start)}</span><div><p>{text}</p></div></li>)}</ol>
      </details>
    }
    if (m.kind === 'script') {
      const lines: { narration?: string; beat?: string }[] = d.lines ?? []
      return <details className="card script-card"><summary>Read the script <b>{lines.length} lines</b></summary>
        <ol className="board">{lines.map((l, i) => <li key={i}><span className="board-n">{i + 1}</span><div><p>{l.narration || <em>(silent) {l.beat}</em>}</p></div></li>)}</ol>
      </details>
    }
    if (m.kind === 'progress' && m.id === last.progress && progressCard && prod.job) {
      const pct = Math.round(100 * (prod.job.progress ?? 0))
      return <div className="card progress-card">
        <div className="progress-head"><span className={`live-dot ${rendering ? '' : 'idle'}`} /><strong>{rendering ? 'Rendering' : prod.job.status}</strong><em>{pct}%</em></div>
        <div className="bar"><i style={{ width: `${pct}%` }} /></div>
        <small>{prod.job.message || prod.job.status}</small>
      </div>
    }
    if (m.kind === 'video' && d.url) {
      return <div className="card video-card">
        <video controls playsInline preload="metadata" src={d.url} poster={d.job_id ? `/outputs/${d.job_id}/thumb.jpg` : undefined} />
        <div className="card-actions">
          {d.job_id && <a className="ghost-button" href={`/edit/${d.job_id}`}>✎ Open in editor</a>}
          <a className="ghost-button" href={String(d.url).split('?')[0]} download>↓ Download</a>
        </div>
      </div>
    }
    if (m.kind === 'stopped' && d.render && m.id === last.stopped && !rendering && !prod.busy && prod.job && prod.job.status !== 'done') {
      return <div className="card-actions"><Button primary onClick={resume}>Resume render <span>▶</span></Button><span className="card-hint">continues from the last finished shot</span></div>
    }
    if (m.kind === 'report' && Array.isArray(d.bad) && d.bad.length) {
      return <div className="card report-card">
        <div className="card-title">QA report <b>{d.bad.length} to reshoot</b></div>
        <table className="kv"><tbody>{d.bad.map((b: { shot: string; problem: string }, i: number) => <tr key={i}><th>Line {parseInt(b.shot.match(/\d+/)?.[0] ?? '0')}</th><td>{b.problem}</td></tr>)}</tbody></table>
        {m.id === last.report && prod.stage === 'done' && !prod.busy && <div className="card-actions"><Button primary onClick={() => say('Please reshoot the shots the QA Critic flagged, fixing exactly those problems.')}>Reshoot these <span>↻</span></Button></div>}
      </div>
    }
    return null
  }

  const busy = prod.busy || sending
  return <>
    <div className="chat-header">
      <div className="task-title"><strong title={prod.title}>{prod.title}</strong><small>{ago(prod.created)} · {prod.messages.length} messages · {money(prod.cost)} Claude usage</small></div>
      {(prod.busy || rendering) && <button className="stop-button" onClick={stop} title="Interrupt the crew or the render">■ Stop</button>}
      <span className={`task-pill ${pillOf(prod)}`}>{prod.busy ? 'Working' : stageLabel[prod.stage] ?? prod.stage}</span>
    </div>
    <div className="chat-scroll" key={prod.id} ref={scroller} onWheel={touched} onTouchMove={touched} onKeyDown={touched} onPointerDown={touched}
      onScroll={(event) => { const el = event.currentTarget; if (Date.now() - userInput.current < 1200) nearBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 90 }}>
      <div className="chat-thread">
      {prod.messages.map((m) => m.from === 'user'
        ? <div className="message-bubble" key={m.id}><Avatar member="user" small /><div className="message-body">{m.text && <p>{m.text}</p>}<SentFiles files={m.data?.attachments} /></div></div>
        : <div className={`chat-msg ${m.kind === 'error' ? 'error' : ''}`} key={m.id}>
          <Avatar member={m.from} />
          <div className="chat-msg-body">
            <div className="chat-msg-head"><strong>{who(m)}</strong><em>{new Date(m.ts * 1000).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</em></div>
            {m.text && <Rich chat text={m.text} />}
            {card(m)}
          </div>
        </div>)}
      {activity && <div className="activity-line"><Dots /><span>{activity}</span></div>}
      </div>
    </div>
    <PendingFiles items={files.items} onRemove={files.remove} />
    <div className={`reply-box ${files.dragging ? 'dropping' : ''}`} {...files.dropProps}>
      <AttachButton onFiles={files.add} disabled={prod.busy} />
      <MentionTextarea value={reply} onChange={setReply} rows={1} maxRows={6} disabled={prod.busy}
        placeholder={prod.busy ? 'The crew is working… press ■ Stop to interrupt, or wait for this step to finish' : 'Reply to the crew… (Enter to send, Shift+Enter for a new line, @ to mention a character)'}
        onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); say(reply) } }} />
      <button className="send-button" onClick={() => say(reply)} disabled={busy || files.uploading || (!reply.trim() && !files.ids.length)}>{sending ? 'Sending…' : files.uploading ? 'Uploading…' : <>Reply <span>↗</span></>}</button>
    </div>
  </>
}

function Button({ primary, onClick, children }: { primary?: boolean; onClick: () => Promise<unknown> | void; children: ReactNode }) {
  const [pending, setPending] = useState(false)
  return <button className={primary ? 'send-button' : 'ghost-button'} disabled={pending} onClick={async () => { setPending(true); try { await onClick() } finally { setPending(false) } }}>{children}</button>
}
