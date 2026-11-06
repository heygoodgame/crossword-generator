export type RunKind = 'plan' | 'fill' | 'unlimited'
export type RunStatus = 'running' | 'done' | 'failed' | 'stopped'

export type Run = {
  id: string
  kind: RunKind
  label: string
  pid: number
  /** Batch output directory, relative to the repo root. */
  root: string
  /** Log file, relative to the repo root. */
  log: string
  startedAt: string
  finishedAt?: string
  status: RunStatus
  exitCode?: number
  tail: string[]
  completed?: number
  planned?: number
}

export type Form = {
  through: string
  games: 'all' | 'midi' | 'mini'
  tracks: 'all' | 'hard' | 'easy'
  difficulty: 'easy' | 'hard' | 'starter'
  size: '5' | '7' | '9'
  count: string
}

declare module 'claude-code' {
  interface PluginState {
    'xw-runs': {
      runs: Shaped<Run[]>
      form: Shaped<Form>
      repo: Shaped<string>
      armed: Shaped<string>
      notice: Shaped<string>
    }
  }
}
