// Types and fetch helpers for the RendVid API (src/rendvid/server.py, library_api.py).

export type Stage = 'briefing' | 'awaiting_brief' | 'planning' | 'awaiting_storyboard' | 'rendering' | 'reviewing' | 'done'
export type Kind = 'text' | 'brief' | 'storyboard' | 'script' | 'lyrics' | 'stopped' | 'progress' | 'video' | 'report' | 'error'
export type Message = { id: string; from: string; text: string; kind: Kind; data: Record<string, any>; ts: number }
export type MemberStats = Record<string, { cost: number; runs: number }>
export type ProductionSummary = {
  id: string; title: string; stage: Stage; updated: number; created: number; job_id?: string | null
  busy?: boolean; activity?: string; active?: string[] | null; cost?: number; member_stats?: MemberStats | null
}
export type Job = { id: string; status: string; progress?: number; message?: string; stage?: string }
export type Production = ProductionSummary & { messages: Message[]; job?: Job }
export type Roster = Record<string, { role: string; model: string }>
export type Skill = { key: string; role: string; name: string; path: string; content: string }
export type Memory = { dir: string; team: string; history: string; members: Record<string, string> }
export type Options = { models?: { llm?: boolean; brain?: string; hindi_voice?: boolean; keyframes?: boolean } }

export type AssetKind = 'character' | 'prop' | 'location' | 'style' | 'music'
export type Asset = {
  id: string; kind: AssetKind; name: string; description: string; tags: string[]; voice?: string | null
  cover: string | null; cover_file?: string | null; images: { file: string; url: string; source: 'upload' | 'generated' }[]; audio?: { file: string; url: string }[]
  lora?: { status: 'none' | 'queued' | 'training' | 'ready' | 'failed'; trigger?: string; steps?: number; progress?: number; message?: string }
  generation?: { status: 'idle' | 'queued' | 'running' | 'failed'; progress?: number; message?: string }
  used_in?: string[]; created: number; updated: number
}

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) { super(message); this.status = status }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init)
  const body = await response.json().catch(() => ({}))
  if (!response.ok) throw new ApiError(typeof body.detail === 'string' ? body.detail : response.statusText, response.status)
  return body as T
}
const send = (method: string) => <T>(path: string, body?: unknown) => api<T>(path, { method, headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) })
export const post = send('POST')
export const patch = send('PATCH')
export const del = send('DELETE')

export const money = (value = 0) => value >= 100 ? `$${Math.round(value).toLocaleString()}` : `$${value.toFixed(2)}`
export const ago = (seconds?: number) => {
  if (!seconds) return ''
  const minutes = Math.round((Date.now() / 1000 - seconds) / 60)
  return minutes < 1 ? 'just now' : minutes < 60 ? `${minutes} min ago` : minutes < 1440 ? `${Math.round(minutes / 60)} h ago` : `${Math.round(minutes / 1440)} d ago`
}

export const stageLabel: Record<Stage, string> = {
  briefing: 'Briefing', awaiting_brief: 'Brief to approve', planning: 'Storyboarding',
  awaiting_storyboard: 'Storyboard to approve', rendering: 'Rendering', reviewing: 'QA review', done: 'Ready',
}
export const needsYou = (stage: Stage) => stage === 'awaiting_brief' || stage === 'awaiting_storyboard'
export const pillOf = (p: ProductionSummary) => needsYou(p.stage) ? 'needs' : p.stage === 'done' && !p.busy ? 'done' : 'in-progress'

// Crew members: avatar colour, initials and a one-line job description (roles come from /api/crew/roster).
export const memberMeta: Record<string, { initials: string; color: string; does: string }> = {
  producer: { initials: 'PR', color: 'coral', does: 'Briefs the film, talks to you' },
  writer: { initials: 'WR', color: 'rose', does: 'Writes the narration' },
  director: { initials: 'DI', color: 'violet', does: 'Owns the creative vision' },
  designer: { initials: 'PD', color: 'gold', does: 'Characters, sets and look' },
  dop: { initials: 'CI', color: 'mint', does: 'Designs every shot' },
  supervisor: { initials: 'SS', color: 'blue', does: 'Checks continuity' },
  sound: { initials: 'SD', color: 'teal', does: 'Narrator and music' },
  editor: { initials: 'ED', color: 'peach', does: 'Cut, titles, pacing' },
  critic: { initials: 'QA', color: 'slate', does: 'Inspects every frame' },
}
export const metaOf = (key: string) => memberMeta[key] ?? { initials: key.slice(0, 2).toUpperCase(), color: 'slate', does: '' }

/** Members working right now: the `active` list when the server sends it, else parsed from the activity line. */
export function activeMembers(p: Pick<ProductionSummary, 'active' | 'activity' | 'busy'> | null | undefined, roster: Roster): string[] {
  if (!p) return []
  if (Array.isArray(p.active)) return p.active
  if (!p.busy || !p.activity) return []
  const head = p.activity.split(/ (?:is|are) /)[0]
  return Object.entries(roster).filter(([, m]) => head.includes(m.role)).map(([key]) => key)
}
