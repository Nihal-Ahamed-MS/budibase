import environment from "../environment"

export interface OPAInput {
  user: string
  agentId: string
  tools: string[]
  credentials: Record<string, string>
}

export interface OPADecision {
  allow: boolean
  allowedTools: string[]
}

export async function checkAgentPolicy(input: OPAInput): Promise<OPADecision> {
  const opaUrl = environment.OPA_URL
  if (!opaUrl) {
    return { allow: true, allowedTools: input.tools }
  }

  const response = await fetch(`${opaUrl}/v1/data/budibase/agent`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input }),
  })

  if (!response.ok) {
    throw new Error(`Failed to check ${response.status} ${response.statusText}`)
  }

  const body = (await response.json()) as {
    result?: { allow?: boolean; allowed_tools?: string[] }
  }
  const result = body.result ?? {}

  return {
    allow: result.allow ?? false,
    allowedTools: result.allowed_tools ?? input.tools,
  }
}
