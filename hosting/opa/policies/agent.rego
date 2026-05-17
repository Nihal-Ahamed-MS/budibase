package budibase.agent

import future.keywords.if

default allow := false

# Allow if the user provides the correct access code.
allow if {
	input.credentials.code == "1000"
}

allowed_tools := input.tools
