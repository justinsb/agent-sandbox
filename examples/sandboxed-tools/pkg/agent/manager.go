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
	"time"

	"k8s.io/klog/v2"

	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/llm"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/sessions"
)

// ErrSessionBusy is returned by ManagedSession.RunTurn when the session is
// already processing a turn.
var ErrSessionBusy = errors.New("session is already processing a turn")

// ErrNotChild is returned by Manager methods that act on a session on behalf
// of another when the target is not a child of the acting session.
var ErrNotChild = errors.New("session is not a child of the calling session")

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

// SessionInfo describes a session for a supervising session: its identity
// and metadata, and its current state if it is loaded.
type SessionInfo struct {
	ID       string
	Metadata sessions.Metadata
	// Loaded reports whether the session is loaded in this Manager; State is
	// only meaningful if it is. A session that is not loaded (for example
	// one created before the server restarted) is idle by definition.
	Loaded bool
	State  SessionState
	// MessageCount is the number of messages in the session's transcript,
	// including the system prompt.
	MessageCount int
}

// Children returns the sessions whose recorded parent is parent, sorted by
// ID. It consults the store, so children from before a restart are
// included; those are reported as not loaded.
func (m *Manager) Children(ctx context.Context, parent string) ([]SessionInfo, error) {
	ids, err := m.store.ListSessions(ctx)
	if err != nil {
		return nil, fmt.Errorf("listing sessions: %w", err)
	}
	var children []SessionInfo
	for _, id := range ids {
		metadata, err := m.store.LoadMetadata(ctx, id)
		if err != nil {
			if errors.Is(err, sessions.ErrNotFound) {
				continue // predates metadata, so it has no parent
			}
			return nil, fmt.Errorf("loading metadata of session %q: %w", id, err)
		}
		if metadata.Parent != parent {
			continue
		}
		info, err := m.describe(ctx, id, *metadata)
		if err != nil {
			return nil, err
		}
		children = append(children, *info)
	}
	return children, nil
}

// Describe returns the state of sessionID on behalf of caller, which must be
// its parent (ErrNotChild otherwise): a session may only inspect the
// sessions it spawned.
func (m *Manager) Describe(ctx context.Context, caller, sessionID string) (*SessionInfo, error) {
	metadata, err := m.childMetadata(ctx, caller, sessionID)
	if err != nil {
		return nil, err
	}
	return m.describe(ctx, sessionID, *metadata)
}

// Transcript returns the messages of sessionID on behalf of caller, which
// must be its parent (ErrNotChild otherwise). For a loaded session it is a
// snapshot taken without waiting for a turn in progress, so a parent can
// read along while the child works.
func (m *Manager) Transcript(ctx context.Context, caller, sessionID string) ([]llm.Message, error) {
	if _, err := m.childMetadata(ctx, caller, sessionID); err != nil {
		return nil, err
	}
	if active := m.Get(sessionID); active != nil {
		return active.Session.Messages(), nil
	}
	messages, err := m.store.LoadSession(ctx, sessionID)
	if err != nil && !errors.Is(err, sessions.ErrNotFound) {
		return nil, fmt.Errorf("loading session %q: %w", sessionID, err)
	}
	return messages, nil
}

// Prompt sends input to sessionID on behalf of caller, which must be its
// parent (ErrNotChild otherwise), and returns once the child's turn has
// started; like a spawned child's first turn it runs in the background,
// autonomously. A child that is not loaded (after a restart) is resumed
// first. ErrSessionBusy is returned if the child is mid-turn: prompts are
// not queued, so the parent should check back rather than pile them up.
func (m *Manager) Prompt(ctx context.Context, caller, sessionID, input string) error {
	metadata, err := m.childMetadata(ctx, caller, sessionID)
	if err != nil {
		return err
	}
	child, err := m.Open(ctx, sessionID, OpenOptions{MustExist: true, Metadata: *metadata})
	if err != nil {
		return fmt.Errorf("resuming child session %q: %w", sessionID, err)
	}
	if err := child.RunTurnDetached(m.ctx, input, autonomousEvents{}); err != nil {
		return fmt.Errorf("prompting child session %q: %w", sessionID, err)
	}
	klog.FromContext(ctx).Info("prompted child session", "session", sessionID, "parent", caller)
	return nil
}

// childMetadata loads sessionID's metadata, checking that caller is its
// parent.
func (m *Manager) childMetadata(ctx context.Context, caller, sessionID string) (*sessions.Metadata, error) {
	if err := sessions.ValidateSessionName(sessionID); err != nil {
		return nil, fmt.Errorf("invalid session ID %q: %w", sessionID, err)
	}
	metadata, err := m.store.LoadMetadata(ctx, sessionID)
	if err != nil {
		if errors.Is(err, sessions.ErrNotFound) {
			// Unknown sessions and sessions without metadata look the same
			// to the caller: neither is one of its children.
			return nil, fmt.Errorf("session %q: %w", sessionID, ErrNotChild)
		}
		return nil, fmt.Errorf("loading metadata of session %q: %w", sessionID, err)
	}
	if caller == "" || metadata.Parent != caller {
		return nil, fmt.Errorf("session %q: %w", sessionID, ErrNotChild)
	}
	return metadata, nil
}

func (m *Manager) describe(ctx context.Context, sessionID string, metadata sessions.Metadata) (*SessionInfo, error) {
	info := &SessionInfo{ID: sessionID, Metadata: metadata}
	if active := m.Get(sessionID); active != nil {
		info.Loaded = true
		info.State = active.State()
		// Count without waiting for a running turn: the transcript length is
		// informational.
		info.MessageCount = len(active.Session.Messages())
		return info, nil
	}
	messages, err := m.store.LoadSession(ctx, sessionID)
	if err != nil && !errors.Is(err, sessions.ErrNotFound) {
		return nil, fmt.Errorf("loading session %q: %w", sessionID, err)
	}
	info.MessageCount = len(messages)
	return info, nil
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

	// stateMu guards state, which is updated by the running turn and read
	// by whoever is describing the session.
	stateMu sync.Mutex
	state   SessionState
}

// SessionState is what a session is doing right now and what it did last,
// as observed by the Manager; it is not persisted.
type SessionState struct {
	// Running reports whether a turn is in progress.
	Running bool
	// TurnStartedAt is when the current turn started, if Running.
	TurnStartedAt time.Time
	// CurrentTool is the tool executing right now, if any, and
	// ToolStartedAt when it started.
	CurrentTool   string
	ToolStartedAt time.Time

	// LastActivityAt is when the session last did anything: a turn or tool
	// call starting or finishing. Zero if nothing happened since it was
	// loaded.
	LastActivityAt time.Time
	// LastAssistantMessage is the assistant's most recent final reply.
	LastAssistantMessage string
	// LastError is the error that ended the most recent turn, or "" if it
	// ended normally.
	LastError string
	// CompletedTurns counts the turns that have ended since the session was
	// loaded, successfully or not.
	CompletedTurns int
}

// State returns a snapshot of the session's state.
func (s *ManagedSession) State() SessionState {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	return s.state
}

func (s *ManagedSession) updateState(update func(state *SessionState)) {
	s.stateMu.Lock()
	defer s.stateMu.Unlock()
	update(&s.state)
}

// trackedTurn runs one turn with the session's harness, recording its
// progress in the session state. The caller holds turnMu.
func (s *ManagedSession) trackedTurn(ctx context.Context, input string, events TurnEvents) error {
	now := time.Now()
	s.updateState(func(state *SessionState) {
		state.Running = true
		state.TurnStartedAt = now
		state.LastActivityAt = now
		state.LastError = ""
	})
	err := s.harness.RunTurn(ctx, s.Session, input, &trackingEvents{session: s, events: events})
	s.updateState(func(state *SessionState) {
		state.Running = false
		state.TurnStartedAt = time.Time{}
		state.CurrentTool = ""
		state.ToolStartedAt = time.Time{}
		state.LastActivityAt = time.Now()
		state.CompletedTurns++
		if err != nil {
			state.LastError = err.Error()
		}
	})
	return err
}

// trackingEvents records a turn's progress in the session state before
// passing each event on.
type trackingEvents struct {
	session *ManagedSession
	events  TurnEvents
}

func (e *trackingEvents) AssistantMessage(ctx context.Context, text string) {
	e.session.updateState(func(state *SessionState) {
		state.LastAssistantMessage = text
		state.LastActivityAt = time.Now()
	})
	e.events.AssistantMessage(ctx, text)
}

func (e *trackingEvents) ApproveToolCall(ctx context.Context, call llm.ToolCall) (bool, error) {
	return e.events.ApproveToolCall(ctx, call)
}

func (e *trackingEvents) ToolCallStarted(ctx context.Context, call llm.ToolCall) {
	e.session.updateState(func(state *SessionState) {
		state.CurrentTool = call.Function.Name
		state.ToolStartedAt = time.Now()
		state.LastActivityAt = state.ToolStartedAt
	})
	e.events.ToolCallStarted(ctx, call)
}

func (e *trackingEvents) ToolCallFinished(ctx context.Context, call llm.ToolCall, err error) {
	e.session.updateState(func(state *SessionState) {
		state.CurrentTool = ""
		state.ToolStartedAt = time.Time{}
		state.LastActivityAt = time.Now()
	})
	e.events.ToolCallFinished(ctx, call, err)
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
	return s.trackedTurn(ctx, input, events)
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
		if err := s.trackedTurn(ctx, input, events); err != nil {
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
