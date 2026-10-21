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

// Package supervisor is a toolset for a session that supervises other
// sessions: it creates child sessions (each with its own toolset and
// sandbox) and delegates work to them, rather than doing the work itself.
//
// The toolset acts on sessions through an agent.Manager, so it is only
// available in front ends that have one (the ACP server), not in the
// single-session CLI; that is also why it is constructed with New(manager)
// rather than by toolsets.Get.
//
// TODO: the agent loop creates the calling session's sandbox before any
// tool call, including ones that only act on sessions; a supervisor that
// never runs a command still gets a sandbox.
package supervisor

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/agent"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/llm"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/tools"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/toolsets/geminicli"
)

// Name is the toolset's name, as selected with TOOLSET / -toolset.
const Name = "supervisor"

// DefaultChildToolset is the toolset child sessions run with unless the
// supervisor asks for another one.
const DefaultChildToolset = "geminicli"

// systemPrompt seeds every new supervisor session.
const systemPrompt = `You are a supervisor agent. You do not do engineering work yourself: you delegate it to child sessions, each of which is an autonomous coding agent with its own sandbox, and you coordinate them.

Use spawn_child_session to start a child on a task. Give it a complete, self-contained prompt: everything it needs to know about the task, since it cannot see this conversation. The call returns the child's session ID as soon as the child has started; the child then works on its own, approving its own tool calls.

Use list_child_sessions to see the children you have started and describe_session to check on one: whether it is still working (and on which tool), how long it has been idle, and its last reply. Use read_session to read a child's transcript when you need to understand what it did or why it is stuck.

Report to the user which children you started and what each one is doing.`

// Toolset is the supervisor toolset.
type Toolset struct {
	manager *agent.Manager
}

// New returns the supervisor toolset, acting on sessions through manager.
func New(manager *agent.Manager) *Toolset {
	return &Toolset{manager: manager}
}

// Name implements toolsets.Toolset.
func (t *Toolset) Name() string {
	return Name
}

// SystemPrompt implements toolsets.Toolset.
func (t *Toolset) SystemPrompt() string {
	return systemPrompt
}

// RegisterTools implements toolsets.Toolset.
func (t *Toolset) RegisterTools(registry *tools.Registry) {
	registry.Add(&SpawnChildSessionTool{manager: t.manager})
	registry.Add(&ListChildSessionsTool{manager: t.manager})
	registry.Add(&DescribeSessionTool{manager: t.manager})
	registry.Add(&ReadSessionTool{manager: t.manager})
}

// DefaultImage implements toolsets.Toolset. The supervisor runs no tools in
// its sandbox yet, so it shares the geminicli image rather than needing one
// of its own.
func (t *Toolset) DefaultImage() string {
	return geminicli.DefaultImage
}

// SpawnChildSessionToolName is the name of the spawn_child_session tool.
const SpawnChildSessionToolName = "spawn_child_session"

// SpawnChildSessionTool creates a child of the calling session and starts it
// on an initial prompt.
type SpawnChildSessionTool struct {
	// Prompt is the child's first user message.
	Prompt string `json:"prompt"`
	// Toolset is the child's toolset; empty means DefaultChildToolset.
	Toolset string `json:"toolset,omitempty"`

	manager *agent.Manager
}

func (t *SpawnChildSessionTool) Schema() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.ToolFunction{
			Name: SpawnChildSessionToolName,
			Description: "Creates a new child session, an autonomous agent with its own sandbox, and sends it an initial prompt. " +
				"Returns the child's session ID immediately; the child then works on the prompt in the background, approving its own tool calls. " +
				"The child cannot see this conversation, so the prompt must contain everything it needs.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"prompt": map[string]any{
						"type":        "string",
						"description": "The task for the child session, as a complete, self-contained prompt.",
					},
					"toolset": map[string]any{
						"type":        "string",
						"description": fmt.Sprintf("The toolset (agent profile) the child runs with. Defaults to %q, an autonomous coding agent.", DefaultChildToolset),
					},
				},
				"required": []string{"prompt"},
			},
		},
	}
}

func (t *SpawnChildSessionTool) Run(ctx context.Context, _ tools.Sandbox) (llm.Message, error) {
	if t.manager == nil {
		return llm.Message{}, errors.New("spawn_child_session is not configured with a session manager")
	}
	if t.Prompt == "" {
		return llm.Message{}, errors.New("prompt is required")
	}
	parent := tools.CallerSession(ctx)
	if parent == "" {
		return llm.Message{}, errors.New("spawn_child_session must be called from within a session")
	}
	toolset := t.Toolset
	if toolset == "" {
		toolset = DefaultChildToolset
	}

	childID, err := t.manager.Spawn(ctx, parent, toolset, t.Prompt)
	if err != nil {
		return llm.Message{}, err
	}

	content := fmt.Sprintf("Spawned child session %s (toolset %s). It is now working on the prompt.", childID, toolset)
	return llm.Message{Content: &content}, nil
}

// ListChildSessionsToolName is the name of the list_child_sessions tool.
const ListChildSessionsToolName = "list_child_sessions"

// ListChildSessionsTool lists the sessions the calling session spawned.
type ListChildSessionsTool struct {
	manager *agent.Manager
}

func (t *ListChildSessionsTool) Schema() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.ToolFunction{
			Name:        ListChildSessionsToolName,
			Description: "Lists the child sessions this session has spawned, one per line with its toolset, state and age. Use describe_session for details on one of them.",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{},
			},
		},
	}
}

func (t *ListChildSessionsTool) Run(ctx context.Context, _ tools.Sandbox) (llm.Message, error) {
	if t.manager == nil {
		return llm.Message{}, errors.New("list_child_sessions is not configured with a session manager")
	}
	caller := tools.CallerSession(ctx)
	if caller == "" {
		return llm.Message{}, errors.New("list_child_sessions must be called from within a session")
	}

	children, err := t.manager.Children(ctx, caller)
	if err != nil {
		return llm.Message{}, err
	}

	var b strings.Builder
	if len(children) == 0 {
		b.WriteString("No child sessions.")
	}
	now := time.Now()
	for _, child := range children {
		fmt.Fprintf(&b, "%s: toolset=%s state=%s created=%s ago\n", child.ID, child.Metadata.Toolset, stateSummary(child, now), formatAge(now.Sub(child.Metadata.CreatedAt)))
	}
	content := strings.TrimRight(b.String(), "\n")
	return llm.Message{Content: &content}, nil
}

// DescribeSessionToolName is the name of the describe_session tool.
const DescribeSessionToolName = "describe_session"

// DescribeSessionTool reports the state of one child session.
type DescribeSessionTool struct {
	// SessionID is the child session to describe.
	SessionID string `json:"session_id"`

	manager *agent.Manager
}

func (t *DescribeSessionTool) Schema() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.ToolFunction{
			Name: DescribeSessionToolName,
			Description: "Describes one of this session's child sessions: whether it is working or idle, which tool it is running, how long it has been idle, " +
				"its last reply and the error that ended its last turn, if any. Only children of this session can be described.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"session_id": map[string]any{
						"type":        "string",
						"description": "The child session's ID, as returned by spawn_child_session or list_child_sessions.",
					},
				},
				"required": []string{"session_id"},
			},
		},
	}
}

func (t *DescribeSessionTool) Run(ctx context.Context, _ tools.Sandbox) (llm.Message, error) {
	if t.manager == nil {
		return llm.Message{}, errors.New("describe_session is not configured with a session manager")
	}
	if t.SessionID == "" {
		return llm.Message{}, errors.New("session_id is required")
	}
	caller := tools.CallerSession(ctx)
	if caller == "" {
		return llm.Message{}, errors.New("describe_session must be called from within a session")
	}

	info, err := t.manager.Describe(ctx, caller, t.SessionID)
	if err != nil {
		return llm.Message{}, err
	}

	now := time.Now()
	var b strings.Builder
	fmt.Fprintf(&b, "Session: %s\n", info.ID)
	fmt.Fprintf(&b, "Toolset: %s\n", info.Metadata.Toolset)
	fmt.Fprintf(&b, "Created: %s ago\n", formatAge(now.Sub(info.Metadata.CreatedAt)))
	fmt.Fprintf(&b, "State: %s\n", stateSummary(*info, now))
	fmt.Fprintf(&b, "Messages: %d\n", info.MessageCount)
	if info.Loaded {
		fmt.Fprintf(&b, "Completed turns since loaded: %d\n", info.State.CompletedTurns)
		if info.State.LastError != "" {
			fmt.Fprintf(&b, "Last turn error: %s\n", info.State.LastError)
		}
		if info.State.LastAssistantMessage != "" {
			fmt.Fprintf(&b, "Last reply:\n%s\n", info.State.LastAssistantMessage)
		}
	} else {
		b.WriteString("Not loaded: the session has not run since this server started (use read_session for its transcript).\n")
	}
	content := strings.TrimRight(b.String(), "\n")
	return llm.Message{Content: &content}, nil
}

// stateSummary renders what a session is doing in one phrase.
func stateSummary(info agent.SessionInfo, now time.Time) string {
	if !info.Loaded {
		return "idle (not loaded)"
	}
	state := info.State
	if state.Running {
		if state.CurrentTool != "" {
			return fmt.Sprintf("running tool %s for %s (turn started %s ago)", state.CurrentTool, formatAge(now.Sub(state.ToolStartedAt)), formatAge(now.Sub(state.TurnStartedAt)))
		}
		return fmt.Sprintf("running (waiting for the model; turn started %s ago)", formatAge(now.Sub(state.TurnStartedAt)))
	}
	if state.LastActivityAt.IsZero() {
		return "idle (no activity since loaded)"
	}
	return fmt.Sprintf("idle for %s", formatAge(now.Sub(state.LastActivityAt)))
}

// formatAge renders a duration coarsely (seconds, minutes, or hours), which
// is all the model needs to decide whether a session is stuck.
func formatAge(d time.Duration) string {
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm", int(d.Minutes()))
	default:
		return fmt.Sprintf("%dh%02dm", int(d.Hours()), int(d.Minutes())%60)
	}
}
