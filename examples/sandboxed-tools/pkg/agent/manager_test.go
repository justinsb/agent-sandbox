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

package agent

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"

	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/llm"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/sessions"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/tools"
)

// blockingLLM is a ChatClient whose completions never return until the
// context is cancelled, so tests can hold a turn open.
type blockingLLM struct {
	started     chan struct{}
	startedOnce sync.Once
}

func (b *blockingLLM) CreateChatCompletion(ctx context.Context, _ llm.ChatCompletionRequest) (*llm.ChatCompletionResponse, error) {
	b.startedOnce.Do(func() { close(b.started) })
	<-ctx.Done()
	return nil, ctx.Err()
}

// silentEvents is a TurnEvents that approves everything and renders nothing.
type silentEvents struct{}

func (silentEvents) AssistantMessage(context.Context, string)                    {}
func (silentEvents) ApproveToolCall(context.Context, llm.ToolCall) (bool, error) { return true, nil }
func (silentEvents) ToolCallStarted(context.Context, llm.ToolCall)               {}
func (silentEvents) ToolCallFinished(context.Context, llm.ToolCall, error)       {}

func newTestManager(t *testing.T, llmClient llm.ChatClient) (*Manager, map[string]*Harness) {
	t.Helper()
	// Sessions snapshot under $HOME; keep the test off the real home directory.
	t.Setenv("HOME", t.TempDir())
	m := NewManager(sessions.NewFileStore(t.TempDir()))
	harnesses := map[string]*Harness{}
	for _, name := range []string{"basic", "geminicli"} {
		h := NewHarness(llmClient, nil, tools.NewRegistry(), RunOptions{Namespace: "ns", HomeDir: "/home/x", SystemPrompt: "You are " + name})
		m.AddToolset(name, h)
		harnesses[name] = h
	}
	return m, harnesses
}

func TestManagerOpen_UsesToolsetHarness(t *testing.T) {
	ctx := t.Context()
	m, harnesses := newTestManager(t, nil)

	if got := strings.Join(m.Toolsets(), ","); got != "basic,geminicli" {
		t.Errorf("Toolsets() = %q, want basic,geminicli", got)
	}

	s, err := m.Open(ctx, "child1", OpenOptions{Metadata: sessions.Metadata{Parent: "p", Toolset: "geminicli"}})
	if err != nil {
		t.Fatalf("Open (new): %v", err)
	}
	if s.harness != harnesses["geminicli"] {
		t.Error("new session should run with the harness of its toolset")
	}
	if s.Session.Metadata.Parent != "p" || s.Session.Metadata.Toolset != "geminicli" {
		t.Errorf("metadata = %+v, want parent p, toolset geminicli", s.Session.Metadata)
	}
	// The system prompt of that toolset seeds the session.
	msgs := s.Messages()
	if len(msgs) != 1 || msgs[0].Role != "system" || *msgs[0].Content != "You are geminicli" {
		t.Errorf("messages = %+v, want the geminicli system prompt", msgs)
	}

	// Opening again returns the same loaded session.
	again, err := m.Open(ctx, "child1", OpenOptions{Metadata: sessions.Metadata{Toolset: "basic"}})
	if err != nil {
		t.Fatalf("Open (again): %v", err)
	}
	if again != s || m.Get("child1") != s {
		t.Error("Open and Get must return the single loaded session")
	}
}

func TestManagerOpen_ResumeKeepsRecordedToolset(t *testing.T) {
	ctx := t.Context()
	m, harnesses := newTestManager(t, nil)

	if _, err := m.Open(ctx, "s1", OpenOptions{Metadata: sessions.Metadata{Toolset: "geminicli"}}); err != nil {
		t.Fatalf("Open: %v", err)
	}

	// A fresh Manager over the same store (as after a restart) resumes the
	// session with its recorded toolset, not the caller's default.
	m2 := NewManager(m.store)
	for name, h := range harnesses {
		m2.AddToolset(name, h)
	}
	s, err := m2.Open(ctx, "s1", OpenOptions{MustExist: true, Metadata: sessions.Metadata{Toolset: "basic"}})
	if err != nil {
		t.Fatalf("Open (resume): %v", err)
	}
	if s.harness != harnesses["geminicli"] {
		t.Error("resumed session should run with its recorded toolset's harness")
	}
	if len(s.Messages()) != 1 {
		t.Errorf("resumed session has %d messages, want 1 (no second system prompt)", len(s.Messages()))
	}
}

func TestManagerOpen_Errors(t *testing.T) {
	ctx := t.Context()
	m, _ := newTestManager(t, nil)

	_, err := m.Open(ctx, "nosuch", OpenOptions{MustExist: true, Metadata: sessions.Metadata{Toolset: "basic"}})
	if !errors.Is(err, sessions.ErrNotFound) {
		t.Errorf("Open(MustExist) err = %v, want ErrNotFound", err)
	}
	if m.Get("nosuch") != nil {
		t.Error("a failed open must not register the session")
	}

	_, err = m.Open(ctx, "s2", OpenOptions{Metadata: sessions.Metadata{Toolset: "supervisor"}})
	if err == nil || !strings.Contains(err.Error(), `toolset "supervisor"`) {
		t.Errorf("Open(unknown toolset) err = %v, want unavailable toolset error", err)
	}

	if _, err := m.Open(ctx, "Not Valid", OpenOptions{Metadata: sessions.Metadata{Toolset: "basic"}}); err == nil {
		t.Error("Open with an invalid session ID should fail")
	}
}

func TestManagedSessionRunTurn_Busy(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	defer cancel()
	llmClient := &blockingLLM{started: make(chan struct{})}
	m, _ := newTestManager(t, llmClient)

	s, err := m.Open(ctx, "s1", OpenOptions{Metadata: sessions.Metadata{Toolset: "basic"}})
	if err != nil {
		t.Fatalf("Open: %v", err)
	}

	done := make(chan error, 1)
	go func() { done <- s.RunTurn(ctx, "hello", silentEvents{}) }()
	<-llmClient.started

	if err := s.RunTurn(ctx, "again", silentEvents{}); !errors.Is(err, ErrSessionBusy) {
		t.Errorf("concurrent RunTurn err = %v, want ErrSessionBusy", err)
	}

	cancel()
	if err := <-done; err == nil {
		t.Error("cancelled turn should fail")
	}
	// The lock is released once the turn ends: a new turn is accepted (and
	// fails only because the context is cancelled, not because it is busy).
	if err := s.RunTurn(ctx, "after", silentEvents{}); errors.Is(err, ErrSessionBusy) {
		t.Error("RunTurn after the turn ended should not be busy")
	}
}
