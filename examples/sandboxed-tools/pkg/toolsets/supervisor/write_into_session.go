// Copyright 2026 The Kubernetes Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package supervisor

import (
	"context"
	"errors"
	"fmt"

	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/agent"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/llm"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/tools"
)

// WriteIntoSessionToolName is the name of the write_into_session tool.
const WriteIntoSessionToolName = "write_into_session"

// WriteIntoSessionTool sends a prompt to a child session.
type WriteIntoSessionTool struct {
	// SessionID is the child session to prompt.
	SessionID string `json:"session_id"`
	// Prompt is the message to send, as the child's next user message.
	Prompt string `json:"prompt"`

	manager *agent.Manager
}

func (t *WriteIntoSessionTool) Schema() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.ToolFunction{
			Name: WriteIntoSessionToolName,
			Description: "Sends a prompt to one of this session's child sessions, as its next user message: a follow-up task, a correction, or an answer to a question in its last reply. " +
				"Returns as soon as the child has started on it; the child works in the background, so check on it later with describe_session or read_session. " +
				"Fails if the child is still busy with its previous prompt; prompts are not queued.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"session_id": map[string]any{
						"type":        "string",
						"description": "The child session's ID.",
					},
					"prompt": map[string]any{
						"type":        "string",
						"description": "The message to send to the child.",
					},
				},
				"required": []string{"session_id", "prompt"},
			},
		},
	}
}

func (t *WriteIntoSessionTool) Run(ctx context.Context, _ tools.Sandbox) (llm.Message, error) {
	if t.manager == nil {
		return llm.Message{}, errors.New("write_into_session is not configured with a session manager")
	}
	if t.SessionID == "" {
		return llm.Message{}, errors.New("session_id is required")
	}
	if t.Prompt == "" {
		return llm.Message{}, errors.New("prompt is required")
	}
	caller := tools.CallerSession(ctx)
	if caller == "" {
		return llm.Message{}, errors.New("write_into_session must be called from within a session")
	}

	if err := t.manager.Prompt(ctx, caller, t.SessionID, t.Prompt); err != nil {
		if errors.Is(err, agent.ErrSessionBusy) {
			return llm.Message{}, fmt.Errorf("session %s is still working on its previous prompt; check on it with describe_session and try again when it is idle", t.SessionID)
		}
		return llm.Message{}, err
	}

	content := fmt.Sprintf("Sent the prompt to session %s. It is now working on it.", t.SessionID)
	return llm.Message{Content: &content}, nil
}
