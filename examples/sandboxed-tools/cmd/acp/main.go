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

package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/acp"
)

// EchoAgent implements a simple echoing ACP agent.
type EchoAgent struct{}

// Initialize returns basic metadata identifying this agent.
func (a *EchoAgent) Initialize(ctx context.Context, clientInfo acp.ClientInfo) (acp.ServerInfo, error) {
	return acp.ServerInfo{
		Name:    "agent-sandbox-acp",
		Version: "0.1.0",
	}, nil
}

// StartSession is called when the client establishes a new session channel.
func (a *EchoAgent) StartSession(ctx context.Context, conn *acp.Connection) (*acp.Session, error) {
	return conn.NewSession(ctx)
}

// Prompt implements the core user greeting echoing flow, utilizing real-time streaming.
func (a *EchoAgent) Prompt(ctx context.Context, session *acp.Session, prompt string) error {
	// 1. Stream reasoning thought chunk
	err := session.StreamThought(ctx, fmt.Sprintf("Formulating greeting echo for: %q", prompt))
	if err != nil {
		return err
	}

	// 2. Stream response message text chunk
	err = session.StreamMessage(ctx, fmt.Sprintf("You said %s", prompt))
	return err
}

func main() {
	acpAddr := ""
	flag.StringVar(&acpAddr, "acp", acpAddr, "Address to serve ACP on (e.g. ws://localhost:3000)")
	flag.Parse()

	if acpAddr == "" {
		fmt.Fprintln(os.Stderr, "Error: --acp flag is required (e.g., --acp ws://localhost:3000)")
		flag.Usage()
		os.Exit(1)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	server := &acp.Server{
		Agent: &EchoAgent{},
	}

	if err := server.ListenAndServe(ctx, acpAddr); err != nil {
		if ctx.Err() != nil {
			fmt.Fprintln(os.Stderr, "Shutting down ACP WebSocket server...")
		} else {
			fmt.Fprintf(os.Stderr, "Server error: %v\n", err)
			os.Exit(1)
		}
	}
}
