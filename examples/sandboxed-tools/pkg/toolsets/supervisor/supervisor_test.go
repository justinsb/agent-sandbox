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
	"time"

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

func TestListAndDescribeSessionTools(t *testing.T) {
	m := newTestManager(t)
	registry := tools.NewRegistry()
	New(m).RegisterTools(registry)
	ctx := tools.WithCallerSession(context.Background(), "parent1")
	call := func(name, args string) (string, error) {
		msg, err := registry.Call(ctx, noSandbox{}, llm.ToolCall{ID: "c", Function: llm.FunctionCall{Name: name, Arguments: args}})
		if err != nil {
			return "", err
		}
		return *msg.Content, nil
	}

	out, err := call(ListChildSessionsToolName, "{}")
	if err != nil || out != "No child sessions." {
		t.Fatalf("list with no children = %q, %v", out, err)
	}

	childID, err := m.Spawn(ctx, "parent1", DefaultChildToolset, "Do the thing")
	if err != nil {
		t.Fatalf("Spawn: %v", err)
	}
	// Someone else's child must not show up or be describable.
	otherID, err := m.Spawn(ctx, "parent2", DefaultChildToolset, "Other thing")
	if err != nil {
		t.Fatalf("Spawn (other): %v", err)
	}
	m.Get(childID).Messages() // wait for the child's turn to finish
	m.Get(otherID).Messages()

	out, err = call(ListChildSessionsToolName, "{}")
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if !strings.Contains(out, childID+": toolset="+DefaultChildToolset+" state=idle for") || strings.Contains(out, otherID) {
		t.Errorf("list = %q, want only %s, idle", out, childID)
	}

	out, err = call(DescribeSessionToolName, `{"session_id":"`+childID+`"}`)
	if err != nil {
		t.Fatalf("describe: %v", err)
	}
	for _, want := range []string{"Session: " + childID, "Toolset: " + DefaultChildToolset, "State: idle for", "Messages: 3", "Completed turns since loaded: 1", "Last reply:"} {
		if !strings.Contains(out, want) {
			t.Errorf("describe = %q, want it to contain %q", out, want)
		}
	}

	for _, args := range []string{`{"session_id":"` + otherID + `"}`, `{"session_id":"nosuch"}`, `{}`} {
		if _, err := call(DescribeSessionToolName, args); err == nil {
			t.Errorf("describe(%s) should fail", args)
		}
	}
}

func TestFormatAge(t *testing.T) {
	for d, want := range map[time.Duration]string{
		5 * time.Second:               "5s",
		3 * time.Minute:               "3m",
		2*time.Hour + 7*time.Minute:   "2h07m",
		25*time.Hour + 30*time.Minute: "25h30m",
	} {
		if got := formatAge(d); got != want {
			t.Errorf("formatAge(%v) = %q, want %q", d, got, want)
		}
	}
}

func TestReadSessionTool(t *testing.T) {
	m := newTestManager(t)
	registry := tools.NewRegistry()
	New(m).RegisterTools(registry)
	ctx := tools.WithCallerSession(context.Background(), "parent1")

	childID, err := m.Spawn(ctx, "parent1", DefaultChildToolset, "What is up?")
	if err != nil {
		t.Fatal(err)
	}
	m.Get(childID).Messages()

	msg, err := registry.Call(ctx, noSandbox{}, llm.ToolCall{ID: "c", Function: llm.FunctionCall{Name: ReadSessionToolName, Arguments: `{"session_id":"` + childID + `"}`}})
	if err != nil {
		t.Fatalf("read_session: %v", err)
	}
	out := *msg.Content
	for _, want := range []string{"Transcript of session " + childID + ": 2 messages", "--- #1 user ---\nWhat is up?", "--- #2 assistant ---"} {
		if !strings.Contains(out, want) {
			t.Errorf("read_session = %q, want it to contain %q", out, want)
		}
	}
	if strings.Contains(out, "system") {
		t.Errorf("read_session = %q, must not include the system prompt", out)
	}

	other, err := m.Spawn(ctx, "parent2", DefaultChildToolset, "x")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := registry.Call(ctx, noSandbox{}, llm.ToolCall{ID: "c", Function: llm.FunctionCall{Name: ReadSessionToolName, Arguments: `{"session_id":"` + other + `"}`}}); err == nil {
		t.Error("reading another session's child should fail")
	}
}

func TestRenderTranscript(t *testing.T) {
	str := func(s string) *string { return &s }
	msgs := []llm.Message{
		{Role: "system", Content: str("secret system prompt")},
		{Role: "user", Content: str("fix it")},
		{Role: "assistant", ToolCalls: []llm.ToolCall{{Function: llm.FunctionCall{Name: "run_shell_command", Arguments: `{"command":"make"}`}}}},
		{Role: "tool", Content: str(strings.Repeat("x", 5000))},
		{Role: "assistant", Content: str("done")},
	}

	out := renderTranscript("s1", msgs, 0)
	if strings.Contains(out, "secret") {
		t.Errorf("system prompt leaked: %q", out)
	}
	for _, want := range []string{"4 messages", "#1 user", "-> run_shell_command({\"command\":\"make\"})", "[3000 more bytes truncated]", "#4 assistant\ndone"} {
		if !strings.Contains(out, want) {
			t.Errorf("render = %q, want %q", out, want)
		}
	}

	out = renderTranscript("s1", msgs, 2)
	if !strings.Contains(out, "showing the last 2; 2 earlier messages omitted") || strings.Contains(out, "#1 user") || !strings.Contains(out, "#3 tool") {
		t.Errorf("render(last 2) = %q, want only the last two messages, numbered from the start", out)
	}

	if got := truncate("héllo", 2); got != "h... [5 more bytes truncated]" {
		t.Errorf("truncate on a rune boundary = %q", got)
	}
}
