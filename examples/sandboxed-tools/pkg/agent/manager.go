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
	"fmt"
	"slices"
	"strings"
	"sync"

	"k8s.io/klog/v2"

	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/llm"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/sessions"
)

// ErrSessionBusy is returned by ManagedSession.RunTurn when the session is
// already processing a turn.
var ErrSessionBusy = errors.New("session is already processing a turn")

// Manager owns the sessions of a long-running front end such as the ACP
// server. It keeps a single in-memory Session per session ID (so that
// several connections loading one ID share it), serializes turns per
// session, and runs each session with the Harness of its toolset, so that
// sessions of different toolsets coexist in one process and a resumed
// session gets the tools its history was recorded against.
//
// Together with the sessions.Store this is the server's session database,
// and it is deliberately simple for an example: process-local, never
// evicted, and persisted only where the store writes. A future iteration
// should consider externalizing it (a shared store with retention/eviction)
// so sessions survive restarts, can be bounded, and can be served by more
// than one replica.
type Manager struct {
	store sessions.Store

	// ctx bounds the turns the Manager runs on its own behalf (child
	// sessions' turns, which no client connection is waiting on); it is the
	// lifetime of the owning server.
	ctx context.Context

	// mu guards harnesses and sessions. It is held across Open's build so
	// that two concurrent opens cannot produce two Session objects for one
	// session ID.
	mu        sync.Mutex
	harnesses map[string]*Harness
	sessions  map[string]*ManagedSession
}

// NewManager creates a Manager whose sessions are persisted in store. ctx
// is the lifetime of the server: it bounds the turns the Manager runs in the
// background for child sessions. Call AddToolset for each toolset sessions
// may run with before opening any.
func NewManager(ctx context.Context, store sessions.Store) *Manager {
	return &Manager{
		store:     store,
		ctx:       ctx,
		harnesses: make(map[string]*Harness),
		sessions:  make(map[string]*ManagedSession),
	}
}

// AddToolset makes sessions of the named toolset runnable, with harness
// providing that toolset's tools, system prompt and sandbox image.
func (m *Manager) AddToolset(toolset string, harness *Harness) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.harnesses[toolset] = harness
}

// Toolsets returns the names of the toolsets sessions can run with, sorted.
func (m *Manager) Toolsets() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.toolsetsLocked()
}

func (m *Manager) toolsetsLocked() []string {
	names := make([]string, 0, len(m.harnesses))
	for name := range m.harnesses {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

// OpenOptions configure Manager.Open.
type OpenOptions struct {
	// MustExist makes opening a session with no persisted history an error
	// wrapping sessions.ErrNotFound, instead of creating the session.
	MustExist bool

	// Metadata describes the session if this call creates it. Its Toolset
	// also serves as the toolset of a resumed session that has none
	// recorded (one from before metadata existed).
	Metadata sessions.Metadata
}

// Open returns the named session, creating or resuming it if it is not
// loaded yet. The session runs with the harness of its recorded toolset
// (or, for a new session, of opts.Metadata.Toolset), which must have been
// registered with AddToolset.
func (m *Manager) Open(ctx context.Context, sessionID string, opts OpenOptions) (*ManagedSession, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if active := m.sessions[sessionID]; active != nil {
		return active, nil
	}

	if opts.MustExist {
		// BuildSession creates-or-resumes, so a load has to check the store
		// itself; otherwise a typo in the ID would silently start a fresh
		// conversation.
		if _, err := m.store.LoadSession(ctx, sessionID); err != nil {
			return nil, err
		}
	}

	toolset := opts.Metadata.Toolset
	stored, err := m.store.LoadMetadata(ctx, sessionID)
	switch {
	case err == nil && stored.Toolset != "":
		toolset = stored.Toolset
	case err == nil, errors.Is(err, sessions.ErrNotFound):
		// No recorded toolset: a new session, or one from before metadata
		// existed; use the caller's.
	default:
		return nil, fmt.Errorf("loading session metadata: %w", err)
	}
	harness := m.harnesses[toolset]
	if harness == nil {
		return nil, fmt.Errorf("session %q uses toolset %q, which is not available here (available: %s)", sessionID, toolset, strings.Join(m.toolsetsLocked(), ", "))
	}

	session, err := harness.BuildSession(ctx, m.store, sessionID, opts.Metadata)
	if err != nil {
		return nil, err
	}
	if err := harness.EnsureSystemPrompt(ctx, session); err != nil {
		return nil, err
	}

	active := &ManagedSession{Session: session, harness: harness}
	m.sessions[sessionID] = active
	return active, nil
}

// Get returns the named session if it is loaded, or nil.
func (m *Manager) Get(sessionID string) *ManagedSession {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.sessions[sessionID]
}

// Spawn creates a new session of the given toolset as a child of the parent
// session and starts it on prompt in the background, returning the child's
// ID as soon as the session exists. The child runs autonomously: its turn is
// not tied to any client connection and its tool calls are approved without
// asking anyone.
func (m *Manager) Spawn(ctx context.Context, parent, toolset, prompt string) (string, error) {
	if err := sessions.ValidateSessionName(parent); err != nil {
		return "", fmt.Errorf("invalid parent session name %q: %w", parent, err)
	}
	childID, err := sessions.NewSessionID("child")
	if err != nil {
		return "", err
	}
	child, err := m.Open(ctx, childID, OpenOptions{Metadata: sessions.Metadata{Parent: parent, Toolset: toolset}})
	if err != nil {
		return "", fmt.Errorf("creating child session: %w", err)
	}
	klog.FromContext(ctx).Info("spawned child session", "session", childID, "parent", parent, "toolset", toolset)

	if err := child.RunTurnDetached(m.ctx, prompt, autonomousEvents{}); err != nil {
		return "", fmt.Errorf("starting child session %q: %w", childID, err)
	}
	return childID, nil
}

// autonomousEvents drives a turn that no one is watching: tool calls are
// approved, and the outcome is only logged.
type autonomousEvents struct{}

func (autonomousEvents) AssistantMessage(ctx context.Context, text string) {
	klog.FromContext(ctx).V(1).Info("assistant message", "text", text)
}

func (autonomousEvents) ApproveToolCall(context.Context, llm.ToolCall) (bool, error) {
	return true, nil
}

func (autonomousEvents) ToolCallStarted(context.Context, llm.ToolCall) {}

func (autonomousEvents) ToolCallFinished(context.Context, llm.ToolCall, error) {}

// ManagedSession is a session loaded in a Manager. Session and
// Harness.RunTurn are not safe for concurrent use, while front ends dispatch
// requests concurrently, so turnMu serializes access to the session.
type ManagedSession struct {
	Session *Session

	harness *Harness
	turnMu  sync.Mutex
}

// RunTurn processes one user message with the session's harness (see
// Harness.RunTurn). Only one turn runs at a time per session; a concurrent
// call fails with ErrSessionBusy rather than queueing, so the caller can
// give an immediate, explicit answer.
func (s *ManagedSession) RunTurn(ctx context.Context, input string, events TurnEvents) error {
	if !s.turnMu.TryLock() {
		return ErrSessionBusy
	}
	defer s.turnMu.Unlock()
	return s.harness.RunTurn(ctx, s.Session, input, events)
}

// RunTurnDetached starts a turn in the background and returns once it has
// started, or ErrSessionBusy if a turn is already in progress. The turn runs
// under ctx (which should outlive the caller, e.g. the server's lifetime);
// its error, if any, is logged rather than returned.
func (s *ManagedSession) RunTurnDetached(ctx context.Context, input string, events TurnEvents) error {
	if !s.turnMu.TryLock() {
		return ErrSessionBusy
	}
	sessionName := s.Session.Name
	go func() {
		defer s.turnMu.Unlock()
		if err := s.harness.RunTurn(ctx, s.Session, input, events); err != nil {
			klog.FromContext(ctx).Error(err, "detached turn failed", "session", sessionName)
		}
	}()
	return nil
}

// Messages returns a copy of the session's messages, waiting for a turn in
// progress to finish first so the copy is consistent.
func (s *ManagedSession) Messages() []llm.Message {
	s.turnMu.Lock()
	defer s.turnMu.Unlock()
	return s.Session.Messages()
}
