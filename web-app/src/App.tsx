import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'
import {
  activeMembers, ago, api, metaOf, money, needsYou, pillOf, post, stageLabel,
  type Memory, type Options, type Production, type ProductionSummary, type Roster, type Skill,
} from './api.ts'
import { Conversation } from './Conversation.tsx'
import { LibraryView } from './Library.tsx'
import { Avatar, Collapse, MentionTextarea, Rich } from './ui.tsx'
import { AttachButton, PendingFiles, useAttachments } from './Attachments.tsx'

type Page = 'command' | 'library'
type Tab = 'films' | 'skills' | 'memory'
type Appearance = 'auto' | 'light' | 'dark'

const examples = [
  ['Music video 🎵', 'Attach your song (📎) and write: a music video with @Roma at @VillageMainHouse, walking around the house, singing, sitting on the well'],
  ['Hindi children\'s story', 'A 2-minute Hindi children\'s story about a clever crow and a thirsty pot'],
  ['Cinematic visual piece', 'A 1-minute cinematic visual piece: Saturn with icy rings, slow dolly-in, NASA documentary style'],
  ['Documentary', 'A 3-minute English documentary about how monsoons shape Indian farming'],
] as const
const noteCount = (text?: string) => (text ?? '').split('\n').filter((line) => /^\s*[-*] /.test(line)).length
const live = (p?: Production | null) => !!p && (!!p.busy || (!!p.job && ['running', 'queued'].includes(p.job.status)))

function useAppearance() {
  const [appearance, setAppearance] = useState<Appearance>(() => { try { return (localStorage.getItem('rendvid.appearance') as Appearance) || 'auto' } catch { return 'auto' } })
  useEffect(() => {
    const media = matchMedia('(prefers-color-scheme: dark)')
    const apply = () => { document.documentElement.dataset.theme = appearance === 'auto' ? (media.matches ? 'dark' : 'light') : appearance }
    apply()
    try { localStorage.setItem('rendvid.appearance', appearance) } catch { /* private mode */ }
    media.addEventListener('change', apply)
    return () => media.removeEventListener('change', apply)
  }, [appearance])
  return [appearance, () => setAppearance((current) => current === 'auto' ? 'light' : current === 'light' ? 'dark' : 'auto')] as const
}

function App() {
  const [page, setPage] = useState<Page>(() => location.hash === '#library' ? 'library' : 'command')
  const [tab, setTab] = useState<Tab>('films')
  const [roster, setRoster] = useState<Roster>({})
  const [films, setFilms] = useState<ProductionSummary[] | null>(null)
  const [selectedId, setSelectedId] = useState<string | null>(() => location.hash.startsWith('#film=') ? location.hash.slice(6) : null)
  const [prod, setProd] = useState<Production | null>(null)
  const [skills, setSkills] = useState<Skill[]>([])
  const [memory, setMemory] = useState<Memory | null>(null)
  const [options, setOptions] = useState<Options | null>(null)
  const [offline, setOffline] = useState(false)
  const [request, setRequest] = useState('')
  const files = useAttachments()
  const [sending, setSending] = useState(false)
  const [notice, setNotice] = useState('')
  const [openSkill, setOpenSkill] = useState<string | null>(null)
  const [openMemory, setOpenMemory] = useState<string | null>(null)
  const [libraryCount, setLibraryCount] = useState<number | null>(null)
  const [appearance, cycleAppearance] = useAppearance()
  const pollTimer = useRef<number | undefined>(undefined)
  const openId = useRef(selectedId)
  openId.current = selectedId

  const loadFilms = useCallback(() => api<ProductionSummary[]>('/api/crew/productions').then((list) => { setFilms(list); setOffline(false) }).catch(() => setOffline(true)), [])
  const loadSide = useCallback(() => {
    api<Skill[]>('/api/crew/skills').then(setSkills).catch(() => {})
    api<Memory>('/api/crew/memory').then(setMemory).catch(() => {})
  }, [])
  useEffect(() => {
    api<Roster>('/api/crew/roster').then(setRoster).catch(() => setOffline(true))
    api<Options>('/api/options').then(setOptions).catch(() => {})
    api<unknown[]>('/api/library').then((list) => setLibraryCount(list.length)).catch(() => setLibraryCount(null))
    loadFilms(); loadSide()
  }, [loadFilms, loadSide])
  const anyBusy = !!films?.some((film) => film.busy)
  useEffect(() => {
    const timer = setInterval(loadFilms, anyBusy ? 3000 : 8000)
    return () => clearInterval(timer)
  }, [loadFilms, anyBusy])
  useEffect(() => { if (tab !== 'films') loadSide() }, [tab, loadSide])

  // The open production: fast polling while the crew or the renderer is working, slow otherwise.
  const poll = useCallback(async (id: string) => {
    window.clearTimeout(pollTimer.current)
    try {
      const next = await api<Production>(`/api/crew/productions/${id}`)
      if (openId.current !== id) return
      setProd(next)
      pollTimer.current = window.setTimeout(() => poll(id), live(next) ? 1500 : 6000)
    } catch {
      pollTimer.current = window.setTimeout(() => poll(id), 6000)
    }
  }, [])
  useEffect(() => {
    setProd(null)
    if (selectedId) poll(selectedId)
    history.replaceState(null, '', page === 'library' ? '#library' : selectedId ? `#film=${selectedId}` : location.pathname)
    return () => window.clearTimeout(pollTimer.current)
  }, [selectedId, poll]) // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { history.replaceState(null, '', page === 'library' ? '#library' : selectedId ? `#film=${selectedId}` : location.pathname) }, [page, selectedId])

  const createFilm = async () => {
    const text = request.trim()
    if ((!text && !files.ids.length) || sending || files.uploading) return
    setSending(true); setNotice('')
    try {
      const created = await post<Production>('/api/crew/productions', { text, attachments: files.ids })
      setRequest(''); files.clear(); setTab('films'); setSelectedId(created.id); loadFilms()
    } catch (error) { setNotice((error as Error).message) }
    setSending(false)
  }
  const openFilm = (id: string) => { setSelectedId(id); setPage('command') }

  // Who is working: every busy production's active members (the open one is the freshest).
  const working = new Map<string, string>()
  for (const film of films ?? []) if (film.id !== prod?.id) for (const key of activeMembers(film, roster)) working.set(key, film.activity ?? '')
  for (const key of activeMembers(prod, roster)) working.set(key, prod?.activity ?? '')
  const stats = (key: string) => (films ?? []).reduce((total, film) => { const s = film.member_stats?.[key]; return { cost: total.cost + (s?.cost ?? 0), runs: total.runs + (s?.runs ?? 0) } }, { cost: 0, runs: 0 })
  const totalCost = (films ?? []).reduce((total, film) => total + (film.cost ?? 0), 0)
  const waiting = (films ?? []).filter((film) => needsYou(film.stage) && !film.busy)
  const memoryEntries = memory ? [
    { key: 'team', label: 'Team notes', text: memory.team },
    ...Object.entries(memory.members).map(([key, text]) => ({ key, label: `${roster[key]?.role ?? key} · personal notes`, text })),
    { key: 'history', label: 'Production history', text: memory.history },
  ] : []
  const memoryCount = memoryEntries.filter((entry) => entry.key !== 'history').reduce((total, entry) => total + noteCount(entry.text), 0)
  const brain = options?.models?.brain

  return <main className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark">R</span><span>RendVid</span></div>
      <nav>
        <button className={`nav-item ${page === 'command' ? 'active' : ''}`} onClick={() => setPage('command')}><span>◈</span> Command center{waiting.length > 0 && <b className="nav-count hot">{waiting.length}</b>}</button>
        <button className={`nav-item ${page === 'library' ? 'active' : ''}`} onClick={() => setPage('library')}><span>☺</span> <em className="nav-label">Characters &amp; Resources</em>{libraryCount ? <b>{libraryCount}</b> : null}</button>
      </nav>
      <div className="sidebar-label">Team <b>{money(totalCost)}</b></div>
      <div className="team-list">{Object.entries(roster).map(([key, member]) => {
        const doing = working.get(key)
        const s = stats(key)
        return <div className={`team-member ${doing !== undefined ? 'working' : ''}`} key={key} title={doing || `${member.role} · ${member.model}`}>
          <Avatar member={key} small />
          <div className="team-member-info"><div className="agent-name"><strong>{member.role}</strong><span className={`status-dot ${doing !== undefined ? 'working' : ''}`} /></div><small>{doing !== undefined ? 'Working…' : metaOf(key).does}</small></div>
          <span className="member-cost">{money(s.cost)}<small>{s.runs} runs</small></span>
        </div>
      })}</div>
      <div className="sidebar-bottom"><div className="user-avatar">Y</div><div className="sidebar-user"><strong>Yogendra</strong><small>Local studio</small></div>
        <button className="icon-button" onClick={cycleAppearance} title={`Appearance: ${appearance} (click to change)`} aria-label={`Appearance: ${appearance}`}>{appearance === 'dark' ? '☾' : appearance === 'light' ? '☀' : '◐'}</button></div>
    </aside>

    <section className="content">
      <header className="topbar"><div className="breadcrumb">RendVid <span>/</span> {page === 'library' ? 'Characters & Resources' : 'Command center'}</div>
        <div className="top-actions"><span className={`connection ${offline ? 'offline' : options ? (options.models?.llm ? 'connected' : '') : ''}`}><i /> {offline ? 'Server offline' : !options ? 'Connecting…' : options.models?.llm ? `${brain ?? 'Claude'} connected` : 'Claude CLI not found'}</span></div></header>

      {page === 'library' ? <div className="page-scroll"><LibraryView onCount={setLibraryCount} /></div> : <div className="split">
        <div className="pane-left">
          <div className="intro compact"><h1>Hello, Yogendra<span>.</span></h1><p className="eyebrow">{new Date().toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' }).toUpperCase()}</p></div>
          <section className={`request-panel ${files.dragging ? 'dropping' : ''}`} {...files.dropProps}>
            <div className="panel-kicker"><span className="spark">✦</span> GIVE THE CREW A FILM</div>
            <MentionTextarea value={request} onChange={setRequest} rows={2} maxRows={8}
              onKeyDown={(event) => { if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) createFilm() }}
              placeholder="Describe a film, paste a script (Hindi or English), attach a song 🎵, or mention a character with @… (⌘↵ to submit)" />
            <PendingFiles items={files.items} onRemove={files.remove} />
            <div className="request-footer">
              <div className="request-tools"><AttachButton onFiles={files.add} disabled={sending} /><span className="path-input first"><span>⌘↵</span> to submit</span><span className="path-input"><span>@</span> mention a character</span></div>
              <button className="send-button" onClick={createFilm} disabled={sending || files.uploading || (!request.trim() && !files.ids.length)}>{sending ? 'Starting…' : files.uploading ? 'Uploading…' : 'Start film'} <span>↗</span></button>
            </div>
            {notice && <div className="notice">{notice}</div>}
          </section>

          {waiting.length > 0 && <section className="needs-you"><div className="needs-you-head"><span className="needs-dot" />Needs you <b>{waiting.length}</b></div>
            {waiting.map((film) => <div key={film.id} className={`needs-row ${selectedId === film.id ? 'selected' : ''}`} onClick={() => openFilm(film.id)}><strong>{film.title}</strong><small>{film.stage === 'awaiting_brief' ? 'The Producer wants you to approve the brief' : 'The storyboard is ready for your approval'}</small><span className="needs-open">Open ↗</span></div>)}</section>}

          <div className="tabs" role="tablist">{([['films', 'Films', films?.length ?? 0], ['skills', 'Skills', skills.length || Object.keys(roster).length], ['memory', 'Memory', memoryCount]] as const).map(([key, label, count]) =>
            <button key={key} role="tab" aria-selected={tab === key} className={`tab ${tab === key ? 'active' : ''}`} onClick={() => setTab(key)}>{label} <b>{count}</b></button>)}</div>
          <div className="tab-panel" role="tabpanel" key={tab}>
            {tab === 'films' && <>
              <p className="tab-hint">Pick a film to open its conversation</p>
              <section className="task-list">
                {films === null && <div className="task-row"><div className="task-title"><strong>Loading films…</strong></div></div>}
                {films?.length === 0 && <div className="task-row"><div className="task-title"><strong>No films yet</strong><small>Give the crew a film above</small></div></div>}
                {films?.map((film) => <div key={film.id} className={`task-row clickable ${selectedId === film.id ? 'selected' : ''}`} onClick={() => openFilm(film.id)}>
                  <span className="film-thumb">{film.job_id ? <img src={`/outputs/${film.job_id}/thumb.jpg`} alt="" loading="lazy" onError={(event) => { event.currentTarget.style.display = 'none' }} /> : null}<i>▶</i></span>
                  <div className="task-title"><strong title={film.title}>{film.title}</strong><small>{film.busy && film.activity ? film.activity : `${stageLabel[film.stage] ?? film.stage}${film.cost ? ` · ${money(film.cost)}` : ''}`}</small></div>
                  <span className={`task-pill ${pillOf(film)}`}>{film.busy ? 'Working' : stageLabel[film.stage] ?? film.stage}</span>
                  <span className="task-time">{ago(film.updated)}</span>
                </div>)}
              </section>
            </>}
            {tab === 'skills' && <>
              <p className="tab-hint">How each role works · added to that member's instructions on every run · edit the files in outputs/.crew/skills to change them</p>
              <section className="task-list">{skills.map((skill) => <div key={skill.key}>
                <div className="task-row clickable" onClick={() => setOpenSkill(openSkill === skill.key ? null : skill.key)}><Avatar member={skill.key} small /><div className="task-title"><strong>{skill.role}</strong><small>{skill.name}.md · {skill.content.split('\n').find((line) => line.startsWith('# '))?.slice(2) ?? metaOf(skill.key).does}</small></div><span className="task-status">{openSkill === skill.key ? '−' : '+'}</span></div>
                <Collapse open={openSkill === skill.key}><div className="task-detail skill-detail">{skill.content ? <Rich className="skill-text" text={skill.content} /> : <p className="tab-hint">No skill file yet.</p>}</div></Collapse>
              </div>)}</section>
            </>}
            {tab === 'memory' && <>
              <p className="tab-hint">What the crew has learned · saved in {memory?.dir ?? 'outputs/.crew'}</p>
              <section className="task-list">{memoryEntries.map((entry) => <div key={entry.key}>
                <div className="task-row clickable" onClick={() => setOpenMemory(openMemory === entry.key ? null : entry.key)}>{entry.key === 'team' || entry.key === 'history' ? <span className="task-status">{openMemory === entry.key ? '−' : '+'}</span> : <Avatar member={entry.key} small />}<div className="task-title"><strong>{entry.label}</strong><small>{noteCount(entry.text)} {entry.key === 'history' ? 'productions' : 'notes'}</small></div></div>
                <Collapse open={openMemory === entry.key}><div className="task-detail"><Rich className="memory-text" text={entry.text || '_Nothing yet._'} /></div></Collapse>
              </div>)}</section>
            </>}
          </div>
          <footer className="footer-note"><span className="lock">⌑</span> Local-first studio <span className="footer-separator">•</span> Renders on this Mac, plans with Claude</footer>
        </div>

        <aside className="pane-right">
          {!selectedId ? <div className="chat-empty"><span className="spark">✦</span><strong>No conversation yet</strong><p>Give the crew a film on the left. Their chat shows up here.</p>
            <div className="examples">{examples.map(([label, text]) => <button key={label} onClick={() => setRequest(text)}><small>{label}</small>{text}</button>)}</div></div>
            : !prod ? <div className="chat-empty"><i className="file-spinner" /><p>Opening the conversation…</p></div>
            : <Conversation key={prod.id} prod={prod} roster={roster} onChanged={() => { poll(prod.id); loadFilms() }} onError={(message) => alert(message)} />}
        </aside>
      </div>}
    </section>
  </main>
}

export default App
