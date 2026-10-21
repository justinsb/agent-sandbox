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
	"strings"
	"testing"

	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/agent"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/llm"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/sessions"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/tools"
)

// noSandbox is a tools.Sandbox for tools that never run commands.
type noSandbox struct{}

func (noSandbox) ExecCommand(context.Context, tools.ExecCommandOptions) (*tools.ExecCommandResult, error) {
	panic("the supervisor's session tools must not run commands in the sandbox")
}

// newTestManager returns a Manager with the geminicli and supervisor
// toolsets, backed by the fake LLM so child turns complete on their own.
func newTestManager(t *testing.T) *agent.Manager {
	t.Helper()
	t.Setenv("HOME", t.TempDir())
	m := agent.NewManager(t.Context(), sessions.NewFileStore(t.TempDir()))
	for _, name := range []string{DefaultChildToolset, Name} {
		registry := tools.NewRegistry()
		if name == Name {
			New(m).RegisterTools(registry)
		}
		m.AddToolset(name, agent.NewHarness(&llm.ElizaClient{}, nil, registry, agent.RunOptions{Namespace: "ns", HomeDir: "/home/x", SystemPrompt: "prompt"}))
	}
	return m
}

func TestSpawnChildSessionTool(t *testing.T) {
	m := newTestManager(t)
	registry := tools.NewRegistry()
	New(m).RegisterTools(registry)

	call := llm.ToolCall{ID: "call_1", Function: llm.FunctionCall{Name: SpawnChildSessionToolName, Arguments: `{"prompt":"Fix the bug"}`}}

	// Outside a session there is no parent to attach the child to.
	if _, err := registry.Call(context.Background(), noSandbox{}, call); err == nil || !strings.Contains(err.Error(), "within a session") {
		t.Errorf("Call without a caller session err = %v, want 'within a session'", err)
	}

	ctx := tools.WithCallerSession(context.Background(), "parent1")
	msg, err := registry.Call(ctx, noSandbox{}, call)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if !strings.Contains(*msg.Content, "toolset "+DefaultChildToolset) {
		t.Errorf("result = %q, want the default child toolset", *msg.Content)
	}

	// The result names the child; it is a child of the caller and received
	// the prompt.
	fields := strings.Fields(*msg.Content)
	childID := fields[3]
	child := m.Get(childID)
	if child == nil {
		t.Fatalf("child %q (from %q) is not loaded", childID, *msg.Content)
	}
	if child.Session.Metadata.Parent != "parent1" {
		t.Errorf("child parent = %q, want parent1", child.Session.Metadata.Parent)
	}
	msgs := child.Messages()
	if len(msgs) < 2 || *msgs[1].Content != "Fix the bug" {
		t.Errorf("child transcript = %+v, want the initial prompt as the first user message", msgs)
	}

	for _, args := range []string{`{}`, `{"prompt":"x","toolset":"nosuch"}`} {
		call.Function.Arguments = args
		if _, err := registry.Call(ctx, noSandbox{}, call); err == nil {
			t.Errorf("Call(%s) should fail", args)
		}
	}
}
