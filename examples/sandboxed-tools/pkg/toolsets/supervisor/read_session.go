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
	"strings"

	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/agent"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/llm"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/tools"
)

// ReadSessionToolName is the name of the read_session tool.
const ReadSessionToolName = "read_session"

const (
	// defaultReadSessionMessages is how many of the most recent messages
	// read_session returns unless asked otherwise.
	defaultReadSessionMessages = 30
	// maxReadSessionMessageBytes caps each rendered message, so one huge
	// tool result (a file dump, a build log) cannot crowd out the rest.
	maxReadSessionMessageBytes = 2000
	// maxReadSessionBytes caps the whole rendering; this is the parent's
	// context window being spent.
	maxReadSessionBytes = 32 * 1024
)

// ReadSessionTool returns a child session's transcript.
type ReadSessionTool struct {
	// SessionID is the child session to read.
	SessionID string `json:"session_id"`
	// LastMessages is how many of the most recent messages to return; <= 0
	// means defaultReadSessionMessages.
	LastMessages int `json:"last_messages,omitempty"`

	manager *agent.Manager
}

func (t *ReadSessionTool) Schema() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.ToolFunction{
			Name: ReadSessionToolName,
			Description: "Returns the transcript of one of this session's child sessions: the user prompts it received, its replies, the tool calls it made and their results, most recent last. " +
				"Long tool results are truncated. Use it to understand what a child has done or why it is stuck; use describe_session for a quick status instead.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"session_id": map[string]any{
						"type":        "string",
						"description": "The child session's ID.",
					},
					"last_messages": map[string]any{
						"type":        "integer",
						"description": fmt.Sprintf("How many of the most recent messages to return (default %d).", defaultReadSessionMessages),
					},
				},
				"required": []string{"session_id"},
			},
		},
	}
}

func (t *ReadSessionTool) Run(ctx context.Context, _ tools.Sandbox) (llm.Message, error) {
	if t.manager == nil {
		return llm.Message{}, errors.New("read_session is not configured with a session manager")
	}
	if t.SessionID == "" {
		return llm.Message{}, errors.New("session_id is required")
	}
	caller := tools.CallerSession(ctx)
	if caller == "" {
		return llm.Message{}, errors.New("read_session must be called from within a session")
	}

	messages, err := t.manager.Transcript(ctx, caller, t.SessionID)
	if err != nil {
		return llm.Message{}, err
	}

	content := renderTranscript(t.SessionID, messages, t.LastMessages)
	return llm.Message{Content: &content}, nil
}

// renderTranscript renders the last lastMessages messages of a transcript
// (all but the system prompt count) in a compact text form, truncating long
// messages and the whole rendering.
func renderTranscript(sessionID string, messages []llm.Message, lastMessages int) string {
	if lastMessages <= 0 {
		lastMessages = defaultReadSessionMessages
	}

	// The system prompt is the toolset's, not the conversation's; leave it
	// out so the window of messages is spent on what the child did.
	var conversation []llm.Message
	for _, msg := range messages {
		if msg.Role != "system" {
			conversation = append(conversation, msg)
		}
	}

	var b strings.Builder
	fmt.Fprintf(&b, "Transcript of session %s: %d messages", sessionID, len(conversation))
	omitted := 0
	if len(conversation) > lastMessages {
		omitted = len(conversation) - lastMessages
		conversation = conversation[omitted:]
		fmt.Fprintf(&b, " (showing the last %d; %d earlier messages omitted)", len(conversation), omitted)
	}
	b.WriteString("\n")

	for i, msg := range conversation {
		fmt.Fprintf(&b, "\n--- #%d %s ---\n", omitted+i+1, msg.Role)
		if msg.Content != nil && *msg.Content != "" {
			b.WriteString(truncate(*msg.Content, maxReadSessionMessageBytes))
			b.WriteString("\n")
		}
		for _, call := range msg.ToolCalls {
			fmt.Fprintf(&b, "-> %s(%s)\n", call.Function.Name, truncate(call.Function.Arguments, maxReadSessionMessageBytes))
		}
		if b.Len() > maxReadSessionBytes {
			fmt.Fprintf(&b, "\n[Transcript truncated at %d bytes; ask for fewer messages]\n", maxReadSessionBytes)
			break
		}
	}
	return strings.TrimRight(b.String(), "\n")
}

// truncate shortens s to at most n bytes (on a rune boundary), noting how
// much was dropped.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	cut := n
	for cut > 0 && !isRuneStart(s[cut]) {
		cut--
	}
	return s[:cut] + fmt.Sprintf("... [%d more bytes truncated]", len(s)-cut)
}

func isRuneStart(b byte) bool {
	return b&0xC0 != 0x80
}
