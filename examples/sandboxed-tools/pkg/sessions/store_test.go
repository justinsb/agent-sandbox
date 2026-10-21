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

package sessions

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/llm"
)

func TestFileStore(t *testing.T) {
	ctx := context.Background()
	tempDir, err := os.MkdirTemp("", "sessions-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	store := NewFileStore(tempDir)

	sessionID := "testsession123"

	// 1. Load non-existent session
	msgs, err := store.LoadSession(ctx, sessionID)
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("LoadSession for non-existent session: got err %v, want ErrNotFound", err)
	}
	if len(msgs) != 0 {
		t.Errorf("expected empty messages, got %d", len(msgs))
	}

	// 2. Append a message
	sysPrompt := "You are a helpful assistant."
	msg1 := llm.Message{
		Role:    "system",
		Content: &sysPrompt,
	}
	if err := store.AppendMessages(ctx, sessionID, msg1); err != nil {
		t.Fatalf("AppendMessage failed: %v", err)
	}

	// 3. Load and assert msg1
	msgs, err = store.LoadSession(ctx, sessionID)
	if err != nil {
		t.Fatalf("LoadSession failed: %v", err)
	}
	if diff := cmp.Diff([]llm.Message{msg1}, msgs); diff != "" {
		t.Errorf("messages mismatch (-want +got):\n%s", diff)
	}

	// 4. Append more messages (user and assistant with tool calls)
	userPrompt := "list files"
	msg2 := llm.Message{
		Role:    "user",
		Content: &userPrompt,
	}
	if err := store.AppendMessages(ctx, sessionID, msg2); err != nil {
		t.Fatalf("AppendMessage failed: %v", err)
	}

	toolCall := llm.ToolCall{
		ID:   "call_1",
		Type: "function",
		Function: llm.FunctionCall{
			Name:      "list_files",
			Arguments: `{"path": "."}`,
		},
	}
	msg3 := llm.Message{
		Role:      "assistant",
		ToolCalls: []llm.ToolCall{toolCall},
	}

	toolResult := "file1.txt\nfile2.txt"
	msg4 := llm.Message{
		Role:       "tool",
		ToolCallID: "call_1",
		Content:    &toolResult,
	}
	if err := store.AppendMessages(ctx, sessionID, msg3, msg4); err != nil {
		t.Fatalf("AppendMessage failed: %v", err)
	}

	// 5. Load and assert everything
	msgs, err = store.LoadSession(ctx, sessionID)
	if err != nil {
		t.Fatalf("LoadSession failed: %v", err)
	}
	wantMsgs := []llm.Message{msg1, msg2, msg3, msg4}
	if diff := cmp.Diff(wantMsgs, msgs); diff != "" {
		t.Errorf("messages mismatch (-want +got):\n%s", diff)
	}
}

func TestFileStoreMetadata(t *testing.T) {
	ctx := context.Background()
	store := NewFileStore(t.TempDir())

	// No metadata until it is saved, including for a session that already
	// has messages (one recorded before metadata existed).
	content := "hi"
	if err := store.AppendMessages(ctx, "legacy", llm.Message{Role: "user", Content: &content}); err != nil {
		t.Fatalf("AppendMessages: %v", err)
	}
	for _, id := range []string{"legacy", "missing"} {
		if _, err := store.LoadMetadata(ctx, id); !errors.Is(err, ErrNotFound) {
			t.Errorf("LoadMetadata(%q) err = %v, want ErrNotFound", id, err)
		}
	}

	want := Metadata{
		Parent:    "parent1",
		Toolset:   "geminicli",
		CreatedAt: time.Date(2026, 10, 21, 18, 0, 0, 0, time.UTC),
	}
	if err := store.SaveMetadata(ctx, "child1", want); err != nil {
		t.Fatalf("SaveMetadata: %v", err)
	}
	got, err := store.LoadMetadata(ctx, "child1")
	if err != nil {
		t.Fatalf("LoadMetadata: %v", err)
	}
	if diff := cmp.Diff(want, *got); diff != "" {
		t.Errorf("metadata mismatch (-want +got):\n%s", diff)
	}

	// Saving metadata must not create a message history.
	if _, err := store.LoadSession(ctx, "child1"); !errors.Is(err, ErrNotFound) {
		t.Errorf("LoadSession after SaveMetadata err = %v, want ErrNotFound", err)
	}

	// Saving again replaces the previous metadata.
	want.Toolset = "basic"
	if err := store.SaveMetadata(ctx, "child1", want); err != nil {
		t.Fatalf("SaveMetadata (replace): %v", err)
	}
	got, err = store.LoadMetadata(ctx, "child1")
	if err != nil {
		t.Fatalf("LoadMetadata (replace): %v", err)
	}
	if got.Toolset != "basic" {
		t.Errorf("Toolset after replace = %q, want basic", got.Toolset)
	}

	if err := store.SaveMetadata(ctx, "Not Valid", want); err == nil {
		t.Error("SaveMetadata with an invalid session name should fail")
	}
}

func TestFileStoreListSessions(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	store := NewFileStore(dir)

	ids, err := store.ListSessions(ctx)
	if err != nil || len(ids) != 0 {
		t.Fatalf("ListSessions(empty) = %v, %v; want none", ids, err)
	}

	content := "hi"
	if err := store.AppendMessages(ctx, "bhistory", llm.Message{Role: "user", Content: &content}); err != nil {
		t.Fatal(err)
	}
	if err := store.SaveMetadata(ctx, "ameta", Metadata{Toolset: "basic"}); err != nil {
		t.Fatal(err)
	}
	// Neither a session: an empty directory, and a file.
	if err := os.Mkdir(dir+"/empty", 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(dir+"/stray.txt", []byte("x"), 0o600); err != nil {
		t.Fatal(err)
	}

	ids, err = store.ListSessions(ctx)
	if err != nil {
		t.Fatalf("ListSessions: %v", err)
	}
	if diff := cmp.Diff([]string{"ameta", "bhistory"}, ids); diff != "" {
		t.Errorf("ListSessions mismatch (-want +got):\n%s", diff)
	}

	if ids, err := NewFileStore(dir + "/does-not-exist").ListSessions(ctx); err != nil || len(ids) != 0 {
		t.Errorf("ListSessions(missing basedir) = %v, %v; want none", ids, err)
	}
}
