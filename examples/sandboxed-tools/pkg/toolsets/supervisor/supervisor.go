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
