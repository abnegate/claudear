import { useState, useEffect, useCallback, useMemo } from 'react'
import useSWR from 'swr'
import {
  fetchConfig, saveConfig, type ConfigResponse,
  fetchGlobalInstruction, saveGlobalInstruction, type InstructionResponse,
} from '../lib/api'
import { PageHeader } from '../components/layout/page-header'
import { CardStackSkeleton } from '../components/shared/page-skeletons'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card'
import { Badge } from '../components/ui/badge'
import {
  Settings, Save, RotateCcw, FileText, AlertTriangle,
  CheckCircle2, Server, GitBranch, Bell, Shield,
  Cpu, Bot, RefreshCw, Eye, X, ShieldAlert,
} from 'lucide-react'
import { useAuth } from '../lib/auth'

interface TomlSection {
  key: string
  label: string
  icon: React.ReactNode
  description: string
  lines: string[]
  startLine: number
}

interface ParsedField {
  key: string
  value: string
  type: 'string' | 'number' | 'boolean' | 'array' | 'unknown'
  comment?: string
  isSecret: boolean
  rawLine: string
}

const SECTION_META: Record<string, { label: string; icon: React.ReactNode; description: string }> = {
  _top: { label: 'Core', icon: <Server className="h-4 w-4 text-primary" />, description: 'Working directory, orgs, and discovery paths' },
  retry: { label: 'Retry Strategy', icon: <RefreshCw className="h-4 w-4 text-orange-500" />, description: 'Retry timing and limits for failed fixes' },
  claude: { label: 'Claude', icon: <Bot className="h-4 w-4 text-violet-500" />, description: 'Model, instructions, and permissions' },
  discord: { label: 'Discord', icon: <Bell className="h-4 w-4 text-indigo-500" />, description: 'Webhook and bot notification settings' },
  ask: { label: 'Human Q&A', icon: <Eye className="h-4 w-4 text-cyan-500" />, description: 'Ask-loop timeouts and semantic reuse' },
  linear: { label: 'Linear', icon: <GitBranch className="h-4 w-4 text-purple-500" />, description: 'Issue source triggers and filters' },
  sentry: { label: 'Sentry', icon: <AlertTriangle className="h-4 w-4 text-red-500" />, description: 'Error tracking source settings' },
  github: { label: 'GitHub', icon: <GitBranch className="h-4 w-4 text-gray-700" />, description: 'Token, PR monitoring, and webhooks' },
  'github.app': { label: 'GitHub App', icon: <Shield className="h-4 w-4 text-gray-500" />, description: 'App-based authentication config' },
  email: { label: 'Email', icon: <Bell className="h-4 w-4 text-green-500" />, description: 'SMTP and IMAP notification settings' },
  sms: { label: 'SMS', icon: <Bell className="h-4 w-4 text-yellow-500" />, description: 'Twilio SMS notification settings' },
  push: { label: 'Push', icon: <Bell className="h-4 w-4 text-pink-500" />, description: 'Pushover notification settings' },
  whatsapp: { label: 'WhatsApp', icon: <Bell className="h-4 w-4 text-green-600" />, description: 'WhatsApp Business Cloud API settings' },
  telegram: { label: 'Telegram', icon: <Bell className="h-4 w-4 text-blue-500" />, description: 'Telegram Bot API settings' },
  regression: { label: 'Regression', icon: <Shield className="h-4 w-4 text-red-400" />, description: 'Post-fix regression monitoring' },
  cascade: { label: 'Cascade', icon: <RefreshCw className="h-4 w-4 text-teal-500" />, description: 'Multi-repo cascade chaining' },
  learning: { label: 'Learning', icon: <Cpu className="h-4 w-4 text-emerald-500" />, description: 'Continuous learning and knowledge extraction' },
}

const SECRET_KEYS = ['token', 'api_key', 'secret', 'password', 'webhook_url', 'private_key', 'auth_token', 'account_sid', 'api_token']

function isSecretKey(key: string): boolean {
  const lower = key.toLowerCase()
  return SECRET_KEYS.some(s => lower.includes(s))
}

function parseFieldType(value: string): 'string' | 'number' | 'boolean' | 'array' | 'unknown' {
  if (value === 'true' || value === 'false') return 'boolean'
  if (value.startsWith('[')) return 'array'
  if (value.startsWith('"') || value.startsWith("'")) return 'string'
  if (/^-?\d+(\.\d+)?$/.test(value)) return 'number'
  return 'unknown'
}

function parseFields(lines: string[]): ParsedField[] {
  const fields: ParsedField[] = []
  let pendingComment = ''

  for (const line of lines) {
    const trimmed = line.trim()
    if (trimmed === '' || trimmed.startsWith('# ==')) {
      pendingComment = ''
      continue
    }
    if (trimmed.startsWith('#') && !trimmed.includes('=')) {
      pendingComment = trimmed.replace(/^#\s*/, '')
      continue
    }
    if (trimmed.startsWith('[')) {
      pendingComment = ''
      continue
    }

    const match = trimmed.match(/^([^=]+?)\s*=\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\[[^\]]*\]|[^#]+?)(?:\s*#.*)?$/)
    if (match) {
      const key = match[1].trim()
      const value = match[2].trim()
      fields.push({
        key,
        value,
        type: parseFieldType(value),
        comment: pendingComment || undefined,
        isSecret: isSecretKey(key),
        rawLine: line,
      })
      pendingComment = ''
    }
  }
  return fields
}

function parseArrayValue(value: string): string[] {
  const inner = value.replace(/^\[/, '').replace(/\]$/, '').trim()
  if (!inner) return []
  const items: string[] = []
  const regex = /"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^,]+/g
  let m: RegExpExecArray | null
  while ((m = regex.exec(inner)) !== null) {
    items.push(m[0].trim().replace(/^["']|["']$/g, ''))
  }
  return items.filter(Boolean)
}

function serializeArrayValue(items: string[]): string {
  return `[${items.map(i => `"${i.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`).join(', ')}]`
}

/** Build a map of sectionKey.fieldKey -> value from parsed TOML content. */
function buildFieldMap(content: string): Record<string, string> {
  const map: Record<string, string> = {}
  const sections = parseSections(content)
  for (const section of sections) {
    const fields = parseFields(section.lines)
    for (const field of fields) {
      map[`${section.key}.${field.key}`] = field.value
    }
  }
  return map
}

function ArrayFieldInput({
  field,
  onChange,
  changed,
}: {
  field: ParsedField
  onChange: (key: string, newValue: string) => void
  changed?: boolean
}) {
  const [newItem, setNewItem] = useState('')
  const items = parseArrayValue(field.value)

  return (
    <div className={`space-y-2 ${changed ? 'rounded-md ring-2 ring-amber-400/50 p-1.5 -m-1.5' : ''}`}>
      <div className="flex flex-wrap gap-1">
        {items.length === 0 && (
          <span className="text-xs text-muted-foreground italic">Empty</span>
        )}
        {items.map((item, i) => (
          <span
            key={i}
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs bg-muted border"
          >
            {item}
            <button
              onClick={() => {
                const next = items.filter((_, j) => j !== i)
                onChange(field.key, serializeArrayValue(next))
              }}
              className="text-muted-foreground hover:text-foreground"
            >
              <X className="h-3 w-3" />
            </button>
          </span>
        ))}
      </div>
      <div className="flex gap-1">
        <input
          type="text"
          value={newItem}
          onChange={e => setNewItem(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && newItem.trim()) {
              e.preventDefault()
              onChange(field.key, serializeArrayValue([...items, newItem.trim()]))
              setNewItem('')
            }
          }}
          placeholder="Add item..."
          className="flex-1 px-2 py-1 text-xs border rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-primary/30"
        />
        <button
          onClick={() => {
            if (newItem.trim()) {
              onChange(field.key, serializeArrayValue([...items, newItem.trim()]))
              setNewItem('')
            }
          }}
          className="px-2 py-1 text-xs border rounded-md hover:bg-muted"
        >
          Add
        </button>
      </div>
    </div>
  )
}

function FieldInput({
  field,
  onChange,
  changed,
}: {
  field: ParsedField
  onChange: (key: string, newValue: string) => void
  changed?: boolean
}) {
  const changedRing = changed ? 'ring-2 ring-amber-400/50' : ''

  if (field.type === 'boolean') {
    const checked = field.value === 'true'
    return (
      <button
        type="button"
        onClick={() => onChange(field.key, checked ? 'false' : 'true')}
        className={`relative w-10 h-5 rounded-full transition-colors cursor-pointer ${
          checked ? 'bg-primary' : 'bg-muted-foreground/30'
        } ${changedRing}`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-4 h-4 rounded-full bg-white transition-transform ${
            checked ? 'translate-x-5' : ''
          }`}
        />
      </button>
    )
  }

  if (field.type === 'number') {
    return (
      <input
        type="number"
        value={field.value}
        onChange={e => onChange(field.key, e.target.value)}
        className={`w-full px-3 py-1.5 text-sm border rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-primary/30 ${changedRing}`}
      />
    )
  }

  if (field.type === 'array') {
    return <ArrayFieldInput field={field} onChange={onChange} changed={changed} />
  }

  // String / unknown
  const displayValue = field.value.replace(/^["']|["']$/g, '').replace(/\\"/g, '"').replace(/\\\\/g, '\\')
  return (
    <input
      type={field.isSecret ? 'password' : 'text'}
      value={displayValue}
      onChange={e => {
        const escaped = e.target.value.replace(/\\/g, '\\\\').replace(/"/g, '\\"')
        onChange(field.key, `"${escaped}"`)
      }}
      className={`w-full px-3 py-1.5 text-sm border rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-primary/30 ${changedRing}`}
    />
  )
}

function parseSections(content: string): TomlSection[] {
  const lines = content.split('\n')
  const sections: TomlSection[] = []
  let currentKey = '_top'
  let currentLines: string[] = []
  let startLine = 0

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const sectionMatch = line.match(/^\[([^\]]+)\]/)
    if (sectionMatch) {
      if (currentLines.length > 0) {
        sections.push({ key: currentKey, lines: currentLines, startLine, ...getMeta(currentKey) })
      }
      currentKey = sectionMatch[1]
      currentLines = [line]
      startLine = i
    } else {
      currentLines.push(line)
    }
  }
  if (currentLines.length > 0) {
    sections.push({ key: currentKey, lines: currentLines, startLine, ...getMeta(currentKey) })
  }

  return sections
}

function getMeta(key: string) {
  const meta = SECTION_META[key]
  if (meta) return meta
  return { label: key, icon: <Settings className="h-4 w-4 text-gray-400" />, description: '' }
}

function SectionFormCard({
  section,
  savedFieldMap,
  onChange,
}: {
  section: TomlSection
  savedFieldMap: Record<string, string>
  onChange: (newLines: string[]) => void
}) {
  const fields = parseFields(section.lines)
  const hasChanges = fields.some(f => {
    const savedValue = savedFieldMap[`${section.key}.${f.key}`]
    return savedValue !== undefined && savedValue !== f.value
  })

  const handleFieldChange = (key: string, newValue: string) => {
    const newLines = section.lines.map(line => {
      const match = line.match(/^(\s*)([^=]+?)\s*=\s*(.+)$/)
      if (match && match[2].trim() === key) {
        return `${match[1]}${key} = ${newValue}`
      }
      return line
    })
    onChange(newLines)
  }

  return (
    <Card className={hasChanges ? 'ring-2 ring-amber-400/40' : ''}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {section.icon}
            <CardTitle className="text-base">{section.label}</CardTitle>
            {hasChanges && (
              <Badge variant="outline" className="text-[10px] text-amber-600 border-amber-300 bg-amber-50">
                Modified
              </Badge>
            )}
          </div>
          <Badge variant="outline" className="text-xs font-mono">
            {section.key === '_top' ? 'root' : `[${section.key}]`}
          </Badge>
        </div>
        {section.description && (
          <CardDescription className="text-xs">{section.description}</CardDescription>
        )}
      </CardHeader>
      <CardContent>
        {fields.length > 0 ? (
          <div className="space-y-3">
            {fields.map(field => {
              const savedValue = savedFieldMap[`${section.key}.${field.key}`]
              const fieldChanged = savedValue !== undefined && savedValue !== field.value
              return (
                <div key={field.key}>
                  <div className="flex items-center gap-2 mb-1">
                    <label className="text-xs font-medium font-mono">{field.key}</label>
                    {field.isSecret && (
                      <span className="text-[10px] text-muted-foreground bg-muted px-1.5 py-0.5 rounded">secret</span>
                    )}
                  </div>
                  {field.comment && (
                    <p className="text-[11px] text-muted-foreground mb-1">{field.comment}</p>
                  )}
                  <FieldInput field={field} onChange={handleFieldChange} changed={fieldChanged} />
                </div>
              )
            })}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground italic">Empty section</p>
        )}
      </CardContent>
    </Card>
  )
}

function GlobalInstructionsCard() {
  const { data, error, isLoading, mutate } = useSWR<InstructionResponse>('global-instruction', fetchGlobalInstruction)
  const [draft, setDraft] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)

  const serverText = data?.text ?? ''
  const text = draft ?? serverText
  const dirty = text !== serverText

  const handleSave = useCallback(async () => {
    const saved = text
    setSaving(true)
    setSaveError(null)
    try {
      await saveGlobalInstruction(saved)
    } catch (e: any) {
      setSaveError(e?.message || 'Failed to save instructions')
      return
    } finally {
      setSaving(false)
    }
    // PUT persisted. Sync the cache optimistically from the known-saved value so
    // a background revalidation failure can't misreport the save or blank the
    // editor. Only clear the draft if the user has not typed more while the PUT
    // was in flight, so newer edits are not discarded.
    setDraft(prev => (prev === saved ? null : prev))
    setSaved(true)
    setTimeout(() => setSaved(false), 3000)
    mutate(prev => (prev ? { ...prev, text: saved } : prev), { revalidate: false })
  }, [text, mutate])

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center gap-2">
          <Bot className="h-4 w-4 text-muted-foreground" />
          <CardTitle className="text-base">Global Agent Instructions</CardTitle>
        </div>
        <CardDescription>
          Prepended to the agent's context on every run, for all repos. Set per-repo
          overrides from the Repos page. This never overwrites a repo's own AGENTS.md.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {error ? (
          <div className="flex items-center gap-2 text-sm text-red-600">
            <AlertTriangle className="h-4 w-4" />
            <span>Failed to load instructions: {error.message}. Editing is disabled to avoid overwriting.</span>
          </div>
        ) : (
          <textarea
            value={text}
            onChange={e => setDraft(e.target.value)}
            disabled={isLoading}
            placeholder="e.g. Prefer minimal diffs. Never edit generated files; fix the generator instead."
            className="w-full font-mono text-xs bg-muted/50 border rounded-md p-4 min-h-[160px] resize-y focus:outline-none focus:ring-2 focus:ring-primary/30 disabled:opacity-50"
            spellCheck={false}
          />
        )}
        <div className="flex items-center gap-3">
          <button
            onClick={handleSave}
            disabled={saving || !dirty || !!error}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-green-600 text-white hover:bg-green-700 disabled:opacity-40 disabled:pointer-events-none transition-colors"
          >
            <Save className="h-3.5 w-3.5" />
            {saving ? 'Saving...' : 'Save Instructions'}
          </button>
          {saved && (
            <span className="text-xs text-green-600 flex items-center gap-1">
              <CheckCircle2 className="h-3.5 w-3.5" /> Saved
            </span>
          )}
          {dirty && !saved && !saveError && <span className="text-xs text-amber-600">Unsaved changes</span>}
          {saveError && (
            <span className="text-xs text-red-600 flex items-center gap-1">
              <AlertTriangle className="h-3.5 w-3.5" /> {saveError}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

export default function ConfigPage() {
  const { user: currentUser } = useAuth()
  const { data, isLoading, error, mutate } = useSWR<ConfigResponse>('config', fetchConfig)
  const [editContent, setEditContent] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [saveResult, setSaveResult] = useState<{ ok: boolean; message: string } | null>(null)
  const [mode, setMode] = useState<'sections' | 'raw'>('sections')

  const savedContent = data?.content ?? ''
  const content = editContent ?? savedContent
  const sections = parseSections(content)
  const hasUnsavedChanges = content !== savedContent

  // Build a field map from the last-saved content for change detection
  const savedFieldMap = useMemo(() => buildFieldMap(savedContent), [savedContent])

  // Initialise edit state from server data on first load
  useEffect(() => {
    if (data?.content && editContent === null) {
      setEditContent(data.content)
    }
  }, [data?.content])  // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (saveResult) {
      const t = setTimeout(() => setSaveResult(null), 5000)
      return () => clearTimeout(t)
    }
  }, [saveResult])

  const handleSave = useCallback(async () => {
    if (!editContent) return
    setSaving(true)
    setSaveResult(null)
    try {
      const result = await saveConfig(editContent)
      setSaveResult(result)
      const freshData = await mutate()
      if (freshData?.content) {
        setEditContent(freshData.content)
      }
    } catch (e: any) {
      setSaveResult({ ok: false, message: e?.message || 'Unknown error' })
    } finally {
      setSaving(false)
    }
  }, [editContent, mutate])

  const handleDiscard = useCallback(() => {
    setEditContent(savedContent)
    setSaveResult(null)
  }, [savedContent])

  const handleSectionChange = useCallback((sectionIdx: number, newLines: string[]) => {
    const current = editContent ?? savedContent
    const allSections = parseSections(current)
    allSections[sectionIdx] = { ...allSections[sectionIdx], lines: newLines }
    const rebuilt = allSections.map(s => s.lines.join('\n')).join('\n')
    setEditContent(rebuilt)
  }, [editContent, savedContent])

  if (currentUser?.role !== 'admin') {
    return (
      <div className="flex items-center gap-2 text-muted-foreground text-sm p-6">
        <ShieldAlert className="h-5 w-5" />
        <span>Access denied. You need administrator privileges to view configuration.</span>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <PageHeader title="Configuration" description="View and edit your claudear.toml config file" />

      <GlobalInstructionsCard />

      {/* Toolbar */}
      <div className="flex items-center gap-3 flex-wrap">
        {data?.path && (
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground bg-muted/50 px-3 py-1.5 rounded-md">
            <FileText className="h-3.5 w-3.5" />
            <span className="font-mono">{data.path}</span>
          </div>
        )}

        {hasUnsavedChanges && (
          <Badge variant="outline" className="text-xs text-amber-600 border-amber-300 bg-amber-50">
            Unsaved changes
          </Badge>
        )}

        <div className="flex-1" />

        {/* View mode toggle */}
        <div className="flex items-center border rounded-md overflow-hidden text-xs">
          <button
            onClick={() => setMode('sections')}
            className={`px-3 py-1.5 transition-colors ${mode === 'sections' ? 'bg-primary text-primary-foreground' : 'bg-card hover:bg-muted'}`}
          >
            Sections
          </button>
          <button
            onClick={() => setMode('raw')}
            className={`px-3 py-1.5 transition-colors ${mode === 'raw' ? 'bg-primary text-primary-foreground' : 'bg-card hover:bg-muted'}`}
          >
            Raw
          </button>
        </div>

        <button
          onClick={handleDiscard}
          disabled={!hasUnsavedChanges}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border hover:bg-muted transition-colors disabled:opacity-40 disabled:pointer-events-none"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Discard
        </button>
        <button
          onClick={handleSave}
          disabled={saving || !hasUnsavedChanges}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-green-600 text-white hover:bg-green-700 disabled:opacity-40 disabled:pointer-events-none transition-colors"
        >
          <Save className="h-3.5 w-3.5" />
          {saving ? 'Saving...' : 'Save'}
        </button>
      </div>

      {/* Save result toast */}
      {saveResult && (
        <div className={`flex items-center gap-2 px-4 py-3 rounded-md text-sm ${
          saveResult.ok
            ? 'bg-green-50 border border-green-200 text-green-800'
            : 'bg-red-50 border border-red-200 text-red-800'
        }`}>
          {saveResult.ok ? <CheckCircle2 className="h-4 w-4" /> : <AlertTriangle className="h-4 w-4" />}
          {saveResult.message}
        </div>
      )}

      {/* Loading */}
      {isLoading && (
        <CardStackSkeleton count={4} itemClassName="h-32" />
      )}

      {/* Error */}
      {error && (
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center gap-2 text-red-600">
              <AlertTriangle className="h-5 w-5" />
              <span className="text-sm">Failed to load config: {error.message}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Section view */}
      {!isLoading && !error && mode === 'sections' && (
        <div className="grid gap-4 md:grid-cols-2">
          {sections.map((section, idx) => (
            <SectionFormCard
              key={section.key + '-' + idx}
              section={section}
              savedFieldMap={savedFieldMap}
              onChange={(newLines) => handleSectionChange(idx, newLines)}
            />
          ))}
        </div>
      )}

      {/* Raw view */}
      {!isLoading && !error && mode === 'raw' && (
        <Card>
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2">
              <FileText className="h-4 w-4 text-muted-foreground" />
              <CardTitle className="text-base">claudear.toml</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            <textarea
              value={content}
              onChange={e => setEditContent(e.target.value)}
              className="w-full font-mono text-xs bg-muted/50 border rounded-md p-4 min-h-[600px] resize-y focus:outline-none focus:ring-2 focus:ring-primary/30"
              spellCheck={false}
            />
          </CardContent>
        </Card>
      )}
    </div>
  )
}
