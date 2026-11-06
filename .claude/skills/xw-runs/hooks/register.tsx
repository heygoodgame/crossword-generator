import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register } from 'claude-code'

import type { Form, Run, RunKind } from '../types'

const PANE = 'xw-runs'
const TITLE = 'Crossword runs'
const STORE_KEY = 'runs-v1'
const MAX_RUNS = 30

const runs = atom({ plugin: 'xw-runs', key: 'runs' } as const, [] as Run[], { shape: 'runs-1' })
const form = atom({ plugin: 'xw-runs', key: 'form' } as const, defaultForm(), { shape: 'form-1' })
const repo = atom({ plugin: 'xw-runs', key: 'repo' } as const, '', { shape: 'repo-1' })
// The action id waiting for a second press (uploads), or ''.
const armed = atom({ plugin: 'xw-runs', key: 'armed' } as const, '', { shape: 'armed-1' })
// One line of feedback under the buttons (validation errors, confirmations).
const notice = atom({ plugin: 'xw-runs', key: 'notice' } as const, '', { shape: 'notice-1' })

function pad(n: number): string {
  return String(n).padStart(2, '0')
}

function isoDate(d: Date): string {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
}

function stamp(): string {
  const d = new Date()
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}`
}

function defaultForm(): Form {
  const through = new Date()
  through.setDate(through.getDate() + 60)
  return {
    through: isoDate(through),
    games: 'all',
    tracks: 'all',
    difficulty: 'easy',
    size: '5',
    count: '50',
  }
}

function sh(arg: string): string {
  return /^[A-Za-z0-9_./=,:-]+$/.test(arg) ? arg : `'${arg.replace(/'/g, `'\\''`)}'`
}

function cli(args: string[]): string {
  return ['uv', 'run', 'crossword-generator', ...args].map(sh).join(' ')
}

function elapsed(fromIso: string, toIso?: string): string {
  const ms = (toIso ? Date.parse(toIso) : Date.now()) - Date.parse(fromIso)
  const min = Math.max(0, Math.round(ms / 60000))
  return min < 60 ? `${min}m` : `${Math.floor(min / 60)}h${pad(min % 60)}m`
}

const GAME_ARGS: Record<Form['games'], string[]> = {
  all: [],
  midi: ['--games', 'midicrossword'],
  mini: ['--games', 'minicrossword'],
}
const TRACK_ARGS: Record<Form['tracks'], string[]> = {
  all: [],
  hard: ['--tracks', 'hard'],
  easy: ['--tracks', 'easy'],
}

// Starts `cmd` detached in its own process group under caffeinate (so a sleeping
// Mac does not pause it) and prints the pid. caffeinate execs the command in
// that pid, so the pid's command line carries the [xw-exit] marker. The run outlives this session; its
// exit status lands as the log's last line.
const LAUNCH = String.raw`cd "$1" && mkdir -p output/batches && set -m
nohup caffeinate -i -s bash -c "$3" > "$2" 2>&1 < /dev/null &
echo $!`

// Args: repeated (id pid log root). Prints "@@ id alive exit", "P key: n" for
// each progress.json found, and "T line" for the last log lines.
const POLL = String.raw`while [ $# -ge 4 ]; do
  id=$1; pid=$2; log=$3; root=$4; shift 4
  alive=0
  if ps -o command= -p "$pid" 2>/dev/null | grep -q 'xw-exit'; then alive=1; fi
  ex=$(grep -a -E '^\[xw-exit\] [0-9]+' "$log" 2>/dev/null | tail -n 1 | cut -d' ' -f2)
  [ -n "$ex" ] || ex=-
  echo "@@ $id $alive $ex"
  if [ -d "$root" ]; then
    find "$root" -maxdepth 3 -name progress.json -exec sh -c 'grep -m 1 -o -E "\"planned\": *[0-9]+" "$1"; grep -m 1 -o -E "\"completed\": *[0-9]+" "$1"' _ {} \; | sed 's/^/P /'
  fi
  tail -n 80 "$log" 2>/dev/null | grep -a -v -E 'httpx|^\[xw-exit\]|^ *$' | tail -n 30 | sed 's/^/T /'
done`

type Poll = { alive: boolean; exit?: number; completed: number; planned: number; tail: string[] }

function parsePoll(stdout: string): Map<string, Poll> {
  const out = new Map<string, Poll>()
  let cur: Poll | undefined
  for (const line of stdout.split('\n')) {
    if (line.startsWith('@@ ')) {
      const [, id = '', alive, ex] = line.split(' ')
      cur = { alive: alive === '1', completed: 0, planned: 0, tail: [] }
      if (ex !== '-') cur.exit = Number(ex)
      out.set(id, cur)
    } else if (cur && line.startsWith('P ')) {
      const m = /"(planned|completed)": *(\d+)/.exec(line)
      if (m) cur[m[1] as 'planned' | 'completed'] += Number(m[2] ?? 0)
    } else if (cur && line.startsWith('T ')) {
      cur.tail.push(line.slice(2))
    }
  }
  return out
}

async function saveRuns($: EngineInterface, fn: (list: Run[]) => Run[]): Promise<Run[]> {
  let next: Run[] = []
  await update($, runs, list => {
    next = fn(list).slice(-MAX_RUNS)
    return next
  })
  await $.store.set(STORE_KEY, next)
  const running = next.filter(r => r.status === 'running').length
  $.ui.status(running > 0 ? `xw: ${running} running` : undefined)
  return next
}

async function poll($: EngineInterface): Promise<void> {
  const root = await read($, repo)
  const list = await read($, runs)
  const live = list.filter(r => r.status === 'running')
  if (!root || live.length === 0) return
  const args = live.flatMap(r => [r.id, String(r.pid), r.log, r.root])
  const res = await $.process.run(['bash', '-c', POLL, 'xw-poll', ...args], { cwd: root, timeoutMs: 20000 })
  const seen = parsePoll(res.stdout)
  const finished: Run[] = []
  await saveRuns($, all =>
    all.map(r => {
      const p = seen.get(r.id)
      if (r.status !== 'running' || !p) return r
      const next: Run = { ...r, tail: p.tail }
      if (p.planned > 0) {
        next.planned = p.planned
        next.completed = p.completed
      }
      if (!p.alive) {
        next.finishedAt = new Date().toISOString()
        next.exitCode = p.exit
        next.status = p.exit === undefined ? 'stopped' : p.exit === 0 ? 'done' : 'failed'
        finished.push(next)
      }
      return next
    }),
  )
  for (const r of finished) {
    const text = `${r.label}: ${r.status}${r.exitCode !== undefined ? ` (exit ${r.exitCode})` : ''}`
    $.ui.toast(text)
    if (r.kind !== 'plan') void $.ui.notify(text, { title: TITLE })
  }
}

async function launch(
  $: EngineInterface,
  kind: RunKind,
  label: string,
  id: string,
  root: string,
  cmd: string,
): Promise<void> {
  const dir = await read($, repo)
  if (!dir) {
    await update($, notice, () => 'Not in the crossword-generator repo; nothing started.')
    return
  }
  const list = await read($, runs)
  const busy = list.find(r => r.status === 'running' && r.kind !== 'plan')
  if (kind !== 'plan' && busy) {
    // Concurrent generator processes can't see each other's grids (the
    // duplicate-grid guard is per process), so run one at a time.
    await update($, notice, () => `Wait for "${busy.label}" to finish (or stop it): parallel runs can ship duplicate grids.`)
    return
  }
  const log = `output/batches/${id}.xw.log`
  const inner = `${cmd}; echo "[xw-exit] $?"`
  const res = await $.process.run(['bash', '-c', LAUNCH, 'xw-launch', dir, log, inner], {
    cwd: dir,
    timeoutMs: 15000,
  })
  const pid = Number.parseInt(res.stdout.trim(), 10)
  if (!Number.isFinite(pid) || res.exitCode !== 0) {
    await update($, notice, () => `Could not start: ${(res.stderr || res.stdout).trim().slice(0, 200)}`)
    return
  }
  const run: Run = {
    id,
    kind,
    label,
    pid,
    root,
    log,
    startedAt: new Date().toISOString(),
    status: 'running',
    tail: [],
  }
  await saveRuns($, all => [...all, run])
  await update($, notice, () => `Started ${label}. Log: ${log}`)
  await update($, armed, () => '')
}

function validDate(s: string): boolean {
  return /^\d{4}-\d{2}-\d{2}$/.test(s) && !Number.isNaN(Date.parse(s))
}

async function startPlan($: EngineInterface): Promise<void> {
  const f = await read($, form)
  if (!validDate(f.through)) return void (await update($, notice, () => 'Through must be YYYY-MM-DD.'))
  const id = `xw-plan-${stamp()}`
  const cmd = cli(['fill-open-days', '--through', f.through, ...GAME_ARGS[f.games], ...TRACK_ARGS[f.tracks], '--plan-only'])
  await launch($, 'plan', `Open days through ${f.through}`, id, `output/batches/${id}`, cmd)
}

async function startFill($: EngineInterface, upload: boolean): Promise<void> {
  const f = await read($, form)
  if (!validDate(f.through)) return void (await update($, notice, () => 'Through must be YYYY-MM-DD.'))
  if (upload && (await read($, armed)) !== 'fill-upload') {
    await update($, armed, () => 'fill-upload')
    await update($, notice, () => 'Press "Fill + upload" again to confirm: every track that passes its gates uploads as draft candidates.')
    return
  }
  const id = `fill-open-days-${stamp()}`
  const root = `output/batches/${id}`
  const scope = [f.games !== 'all' ? f.games : '', f.tracks !== 'all' ? f.tracks : ''].filter(Boolean).join(' ')
  const cmd = cli([
    'fill-open-days',
    '--through', f.through,
    '--output-root', root,
    '--batch-prefix', id,
    ...GAME_ARGS[f.games],
    ...TRACK_ARGS[f.tracks],
    ...(upload ? ['--upload'] : []),
  ])
  const label = `Fill ${scope ? `${scope} ` : ''}open days through ${f.through}${upload ? ' + upload' : ' (dry run)'}`
  await launch($, 'fill', label, id, root, cmd)
}

async function startUnlimited($: EngineInterface): Promise<void> {
  const f = await read($, form)
  const count = Number.parseInt(f.count, 10)
  if (!Number.isInteger(count) || count < 1 || count > 500) {
    return void (await update($, notice, () => 'Count must be a whole number from 1 to 500.'))
  }
  const size = f.difficulty === 'starter' ? '5' : f.size
  const id = `unlimited-${f.difficulty}${size}-${stamp()}`
  const root = `output/batches/${id}`
  const args = [
    'generate-pilot-batch',
    '--output-root', root,
    '--batch-id', id,
    '--buckets', `${f.difficulty}/${size}`,
    '--count', String(count),
    '--no-intra-batch-dedup',
    '--no-exclude-recent-answers',
    '--no-exclude-scheduled-sixty',
    '--avoid-existing-clues',
    '--answer-novelty-candidates', '12',
    '--unlimited-usage-penalty', '2.0',
    '--max-workers', '6',
    '--llm', 'claude',
  ]
  let cmd: string
  if (f.difficulty === 'starter') {
    // Starter list still carries family-unfriendly words; exclude them down to 3 letters.
    const exclude = `output/batches/${id}.starter-exclude.txt`
    cmd =
      `cut -d';' -f1 dictionaries/XwiJeffChenList-NotFamilyFriendly.txt dictionaries/HggGeneratedSafetyExclude.txt | sort -u > ${exclude} && ` +
      cli([...args, '--exclude-answers-file', exclude, '--exclude-answers-min-length', '3'])
  } else {
    cmd = cli(args)
  }
  await launch($, 'unlimited', `Unlimited ${f.difficulty} ${size}x${size} × ${count}`, id, root, cmd)
}

async function stop($: EngineInterface, run: Run): Promise<void> {
  const dir = await read($, repo)
  await $.process.run(['bash', '-c', 'kill -TERM -- "-$1" 2>/dev/null || kill -TERM "$1"', 'xw-stop', String(run.pid)], {
    cwd: dir || undefined,
  })
  await update($, notice, () => `Sent stop to ${run.label}.`)
}

async function review($: EngineInterface, run: Run): Promise<void> {
  const where = `Log: ${run.log}. Output: ${run.root}. Status: ${run.status}${run.exitCode !== undefined ? ` (exit ${run.exitCode})` : ''}.`
  const ask: Record<RunKind, string> = {
    plan: `The fill-open-days plan run "${run.label}" finished. ${where} Summarise the open days per game/track, and check which of them already have pending generated drafts tagged with metadata.target_date so we don't regenerate them.`,
    fill: `The fill-open-days run "${run.label}" finished. ${where} Check each track's gates (check-batch-answers, fill grade_report.passing, the upload dry-run) and tell me what is ready to upload and what needs regenerating. Don't upload anything without asking me.`,
    unlimited: `The unlimited batch "${run.label}" finished. ${where} Verify every result's fill passed grading, run the answer scans, and propose the filtered manifest, upload and publish-unlimited steps. Don't upload or promote without asking me.`,
  }
  await $.prompt.submit({ text: ask[run.kind] })
}

export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    const stored = await $.store.get(STORE_KEY)
    if (Array.isArray(stored)) await update($, runs, () => stored as Run[])

    const top = await $.process.run(['git', 'rev-parse', '--show-toplevel'], { timeoutMs: 10000 }).catch(() => undefined)
    const dir = top && top.exitCode === 0 ? top.stdout.trim() : ''
    if (dir) {
      const py = await $.fs.read(`${dir}/pyproject.toml`).catch(() => '')
      if (typeof py === 'string' && py.includes('crossword-generator')) await update($, repo, () => dir)
    }

    await $.command.register({ name: 'xw', description: 'Open the crossword runs pane' })
    // Open the pane by itself in this repo. An unasked pane waits for a wide
    // enough terminal (about 110+ columns) before it takes a seat.
    if (await read($, repo)) void $.ui.open({ id: PANE, title: TITLE })
    $.clock.every(5000, () => {
      void poll($).catch(() => undefined)
    })
    void poll($).catch(() => undefined)

    return next(e)
  })

  on('command.run', { command: 'xw' }, async $ => {
    await $.ui.open({ id: PANE, title: TITLE })
    return { text: 'Crossword runs pane opened.' }
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    const { Box, Text, Button, Input, Select } = $.ui.resolve(e)
    const dir = await read($, repo)
    const f = await read($, form)
    const list = await read($, runs)
    const msg = await read($, notice)
    const arm = await read($, armed)

    const setForm = (patch: Partial<Form>) => update($, form, cur => ({ ...cur, ...patch }))

    if (!dir) {
      return (
        <Box flexDirection="column">
          <Text color="yellow">Open this session in the crossword-generator repo to use this pane.</Text>
        </Box>
      )
    }

    const shown = [...list].reverse()

    return (
      <Box flexDirection="column" gap={1}>
        <Box flexDirection="column">
          <Text bold>Daily open days</Text>
          <Input
            key="through"
            label="Through "
            value={f.through}
            placeholder="YYYY-MM-DD"
            onInput={(v: string) => void setForm({ through: v.trim() })}
            onSubmit={(v: string) => void setForm({ through: v.trim() })}
          />
          <Box flexDirection="row" gap={2}>
            <Select
              key="games"
              label="Games "
              value={f.games}
              options={[
                { value: 'all', label: 'midi + mini' },
                { value: 'midi', label: 'midi' },
                { value: 'mini', label: 'mini' },
              ]}
              onSelect={(v: string) => void setForm({ games: v as Form['games'] })}
            />
            <Select
              key="tracks"
              label="Tracks "
              value={f.tracks}
              options={[
                { value: 'all', label: 'hard + easy' },
                { value: 'hard', label: 'hard' },
                { value: 'easy', label: 'easy' },
              ]}
              onSelect={(v: string) => void setForm({ tracks: v as Form['tracks'] })}
            />
          </Box>
          <Box flexDirection="row" gap={1}>
            <Button key="plan" hotkey="1" onPress={() => void startPlan($)}>
              Check open days
            </Button>
            <Button key="fill" hotkey="2" onPress={() => void startFill($, false)}>
              Fill (dry run)
            </Button>
            <Button key="fill-upload" hotkey="3" variant={arm === 'fill-upload' ? 'primary' : undefined} onPress={() => void startFill($, true)}>
              {arm === 'fill-upload' ? 'Confirm fill + upload' : 'Fill + upload'}
            </Button>
          </Box>
        </Box>

        <Box flexDirection="column">
          <Text bold>Unlimited pool</Text>
          <Box flexDirection="row" gap={2}>
            <Select
              key="difficulty"
              label="Difficulty "
              value={f.difficulty}
              options={[
                { value: 'easy', label: 'easy' },
                { value: 'hard', label: 'hard' },
                { value: 'starter', label: 'starter (5x5)' },
              ]}
              onSelect={(v: string) => void setForm({ difficulty: v as Form['difficulty'] })}
            />
            <Select
              key="size"
              label="Size "
              value={f.difficulty === 'starter' ? '5' : f.size}
              options={[
                { value: '5', label: '5x5' },
                { value: '7', label: '7x7' },
                { value: '9', label: '9x9' },
              ]}
              onSelect={(v: string) => void setForm({ size: v as Form['size'] })}
            />
          </Box>
          <Input
            key="count"
            label="Count "
            value={f.count}
            onInput={(v: string) => void setForm({ count: v.trim() })}
            onSubmit={(v: string) => void setForm({ count: v.trim() })}
          />
          <Button key="unlimited" hotkey="4" onPress={() => void startUnlimited($)}>
            Generate unlimited batch
          </Button>
          <Text dimColor>Uploading and promoting stay manual: press Review on the finished run.</Text>
        </Box>

        {msg !== '' && <Text color="cyan">{msg}</Text>}

        <Box flexDirection="column">
          <Text bold>Runs</Text>
          {shown.length === 0 && <Text dimColor>No runs yet.</Text>}
          {shown.map(r => {
            const icon = r.status === 'running' ? '●' : r.status === 'done' ? '✓' : r.status === 'failed' ? '✗' : '■'
            const color = r.status === 'running' ? 'yellow' : r.status === 'done' ? 'green' : r.status === 'failed' ? 'red' : 'gray'
            const progress = r.planned ? ` · ${r.completed ?? 0}/${r.planned}` : ''
            const lines = r.kind === 'plan' && r.status !== 'running' ? r.tail.slice(-25) : r.tail.slice(-3)
            return (
              <Box key={r.id} flexDirection="column" marginTop={1}>
                <Text>
                  <Text color={color}>{icon} {r.status}</Text> {r.label}
                  <Text dimColor>
                    {progress} · {elapsed(r.startedAt, r.finishedAt)}
                    {r.exitCode !== undefined && r.exitCode !== 0 ? ` · exit ${r.exitCode}` : ''}
                  </Text>
                </Text>
                {lines.map((line, i) => (
                  <Text key={`${r.id}-t${i}`} dimColor wrap="truncate-end">
                    {'  '}
                    {line}
                  </Text>
                ))}
                <Box flexDirection="row" gap={1}>
                  {r.status === 'running' && (
                    <Button key={`${r.id}-stop`} label="Stop" onPress={() => void stop($, r)} />
                  )}
                  {r.status !== 'running' && (
                    <Button key={`${r.id}-review`} label="Review with Claude" onPress={() => void review($, r)} />
                  )}
                  {r.status !== 'running' && (
                    <Button
                      key={`${r.id}-dismiss`}
                      label="Dismiss"
                      onPress={() => void saveRuns($, all => all.filter(x => x.id !== r.id))}
                    />
                  )}
                </Box>
              </Box>
            )
          })}
        </Box>
      </Box>
    )
  })
}
