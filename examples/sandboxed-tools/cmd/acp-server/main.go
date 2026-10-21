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

// acp-server is the server-side version of the sandboxed-tools example: it
// exposes the agentic loop over the Agent Client Protocol (ACP), speaking
// JSON-RPC 2.0 as newline-delimited JSON over TCP.
//
// It is designed to run inside a Kubernetes cluster (where it creates Agent
// Sandboxes for tool execution) and be reached from a workstation with
// kubectl port-forward; any ACP client can then create a session, send
// prompts, and approve or reject the tool calls the agent wants to run.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"slices"
	"strings"
	"syscall"

	"k8s.io/klog/v2"

	"sigs.k8s.io/agent-sandbox/examples/agentclientprotocol/pkg/acp"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/agent"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/llm"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/sessions"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/tools"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/toolsets"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/toolsets/geminicli"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/toolsets/geminicli/toolbox"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/toolsets/supervisor"
)

func main() {
	ctx := context.Background()

	signalCtx, cancel := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	klog.InitFlags(nil)

	toolsetName := os.Getenv("TOOLSET")

	var opts agent.RunOptions
	opts.InitDefaults()
	listen := flag.String("listen", ":8090", "TCP address to serve ACP on")
	flag.StringVar(&opts.Namespace, "namespace", opts.Namespace, "namespace for sandbox pods")
	flag.StringVar(&opts.Image, "image", opts.Image, "Sandbox container image; if empty, uses the toolset's default image.")
	flag.StringVar(&opts.HomeDir, "homedir", opts.HomeDir, "Home directory in the sandbox; this is currently the only directory that we persist with snapshot/restore.")
	flag.StringVar(&toolsetName, "toolset", toolsetName, fmt.Sprintf("Toolset (tools + system prompt) for sessions created over ACP (a resumed session keeps its recorded toolset); one of: %s", strings.Join(append(toolsets.Names(), supervisor.Name), ", ")))
	flag.DurationVar(&opts.ToolTimeout, "tool-timeout", opts.ToolTimeout, "Maximum duration a single tool invocation may run before it is cancelled. <= 0 disables the timeout.")
	flag.Parse()

	if err := run(signalCtx, opts, toolsetName, *listen); err != nil {
		fmt.Fprintf(os.Stderr, "acp-server: %v\n", err)
		os.Exit(1)
	}
}

func run(ctx context.Context, opts agent.RunOptions, toolsetName string, listen string) error {
	log := klog.FromContext(ctx)

	llmClient, err := llm.NewFromEnv(opts.ModelName)
	if err != nil {
		return fmt.Errorf("failed to initialize llm client: %w", err)
	}

	restConfig, err := agent.GetRESTConfig()
	if err != nil {
		return fmt.Errorf("failed to get kubernetes configuration: %w", err)
	}

	sandboxClient, err := agent.NewSandboxClient(restConfig)
	if err != nil {
		return fmt.Errorf("failed to initialize sandbox client: %w", err)
	}

	// Delete the sandboxes we created so shutdown is prompt rather than
	// waiting for their shutdown time to expire. A connection still mid-turn
	// can create a sandbox after this runs; the shutdown time bounds that.
	defer func() {
		if err := sandboxClient.DeleteAllSandboxes(context.WithoutCancel(ctx)); err != nil {
			log.Error(err, "failed to delete all sandboxes")
		}
	}()

	homeDir, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get user home directory: %w", err)
	}
	// Session history lives on the pod's local filesystem (see agent.Manager
	// for the plan to externalize it).
	sessionStore := sessions.NewFileStore(filepath.Join(homeDir, ".local", "sandboxed-tools"))

	// Every toolset is available, so that a resumed session can run with the
	// one it recorded and a supervisor can spawn children of any toolset;
	// the default toolset is for sessions created over ACP.
	manager := agent.NewManager(ctx, sessionStore)
	for _, name := range toolsets.Names() {
		toolset, err := toolsets.Get(name)
		if err != nil {
			return err
		}
		manager.AddToolset(name, newHarness(ctx, toolset, llmClient, sandboxClient, opts))
	}
	// The supervisor acts on other sessions through the manager, so it is
	// built here rather than by toolsets.Get.
	manager.AddToolset(supervisor.Name, newHarness(ctx, supervisor.New(manager), llmClient, sandboxClient, opts))

	defaultToolset := toolsetName
	if defaultToolset == "" {
		defaultToolset = "basic"
	}
	if !slices.Contains(manager.Toolsets(), defaultToolset) {
		return fmt.Errorf("unknown toolset %q (available: %s)", defaultToolset, strings.Join(manager.Toolsets(), ", "))
	}

	listener, err := net.Listen("tcp", listen)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", listen, err)
	}
	defer listener.Close()
	log.Info("serving ACP", "address", listener.Addr().String(), "model", opts.ModelName, "defaultToolset", defaultToolset)

	// Close the listener on shutdown so Accept returns.
	go func() {
		<-ctx.Done()
		listener.Close()
	}()

	for {
		conn, err := listener.Accept()
		if err != nil {
			if ctx.Err() != nil {
				return nil
			}
			return fmt.Errorf("accept failed: %w", err)
		}
		log.Info("client connected", "remote", conn.RemoteAddr().String())
		go func() {
			defer conn.Close()
			serveConn(ctx, manager, defaultToolset, conn)
			log.Info("client disconnected", "remote", conn.RemoteAddr().String())
		}()
	}
}

// newHarness builds the harness for one toolset: its tools, its system
// prompt, and its sandbox image. Image precedence is as in the CLI: the
// -image flag / SANDBOX_IMAGE env (which then applies to every toolset),
// then the toolset's preferred image, then the generic default.
func newHarness(ctx context.Context, toolset toolsets.Toolset, llmClient llm.ChatClient, sandboxClient *agent.SandboxClient, opts agent.RunOptions) *agent.Harness {
	opts.SystemPrompt = toolset.SystemPrompt()
	if opts.Image == "" {
		opts.Image = toolset.DefaultImage()
	}
	if opts.Image == "" {
		opts.Image = agent.DefaultImage
	}

	toolsRegistry := tools.NewRegistry()
	toolsRegistry.ToolTimeout = opts.ToolTimeout
	toolset.RegisterTools(toolsRegistry)

	klog.FromContext(ctx).Info("toolset available", "toolset", toolset.Name(), "image", opts.Image)
	return agent.NewHarness(llmClient, sandboxClient, toolsRegistry, opts)
}

// serveConn speaks ACP with one client until the connection closes.
//
// acp.Client is a symmetric JSON-RPC peer (calls out, answers incoming
// requests, sends and receives notifications), so we reuse it here with the
// roles reversed: OnRequest serves the client's initialize / session/new /
// session/prompt calls, Notify streams session updates, and Call asks the
// client for tool call permission.
func serveConn(ctx context.Context, manager *agent.Manager, toolsetName string, conn net.Conn) {
	// acp.Client answers each request on its own goroutine and Run returns
	// as soon as the connection closes, without waiting for them. Scope the
	// context to the connection so an in-flight RunTurn (LLM call, sandbox
	// creation, tool exec) is cancelled when the client goes away instead of
	// running on and holding the session's turnMu.
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	peer := acp.NewClient(conn, conn)

	c := &acpConn{
		ctx:         ctx,
		peer:        peer,
		manager:     manager,
		toolsetName: toolsetName,
	}
	peer.OnRequest = c.handleRequest

	peer.Run()
}

// acpConn is the per-connection ACP server state.
type acpConn struct {
	ctx     context.Context
	peer    *acp.Client
	manager *agent.Manager
	// toolsetName is the toolset of sessions created on this connection.
	toolsetName string
}

func (c *acpConn) handleRequest(method string, params json.RawMessage) (any, *acp.RPCError) {
	switch method {
	case "initialize":
		return acp.InitializeResponse{
			ProtocolVersion: acp.ProtocolVersion,
			AgentInfo: &acp.AgentInfo{
				Name:    "sandboxed-tools",
				Version: "0.1.0",
			},
			AgentCapabilities: map[string]any{"loadSession": true},
		}, nil

	case "session/new":
		var req acp.NewSessionRequest
		if err := json.Unmarshal(params, &req); err != nil {
			return nil, invalidParams(err)
		}
		sessionID, err := sessions.NewSessionID("acp")
		if err != nil {
			return nil, internalError(err)
		}
		if _, err := c.manager.Open(c.ctx, sessionID, c.openOptions(false)); err != nil {
			return nil, internalError(err)
		}
		return acp.NewSessionResponse{SessionID: sessionID}, nil

	case "session/load":
		var req acp.LoadSessionRequest
		if err := json.Unmarshal(params, &req); err != nil {
			return nil, invalidParams(err)
		}
		if err := sessions.ValidateSessionName(req.SessionID); err != nil {
			return nil, invalidParams(fmt.Errorf("invalid sessionId: %w", err))
		}
		active, err := c.manager.Open(c.ctx, req.SessionID, c.openOptions(true))
		if err != nil {
			if errors.Is(err, sessions.ErrNotFound) {
				return nil, invalidParams(fmt.Errorf("unknown sessionId %q", req.SessionID))
			}
			return nil, internalError(err)
		}
		c.replayHistory(req.SessionID, active.Messages())
		return struct{}{}, nil

	case "session/prompt":
		var req acp.PromptRequest
		if err := json.Unmarshal(params, &req); err != nil {
			return nil, invalidParams(err)
		}
		return c.prompt(req)

	default:
		return nil, &acp.RPCError{Code: acp.MethodNotFound, Message: fmt.Sprintf("method not supported: %s", method)}
	}
}

// openOptions are the Manager.Open options for sessions on this connection:
// created with (or, if they predate metadata, resumed with) the connection's
// toolset.
func (c *acpConn) openOptions(mustExist bool) agent.OpenOptions {
	return agent.OpenOptions{
		MustExist: mustExist,
		Metadata:  sessions.Metadata{Toolset: c.toolsetName},
	}
}

// replayHistory streams a resumed session's messages to the client, as ACP
// requires for session/load.
func (c *acpConn) replayHistory(sessionID string, messages []llm.Message) {
	for _, msg := range messages {
		if msg.Content == nil || *msg.Content == "" {
			continue
		}
		switch msg.Role {
		case "user":
			c.sendUpdate(sessionID, acp.SessionUpdate{
				SessionUpdateKind: acp.UpdateUserMessageChunk,
				Content:           textContent(*msg.Content),
			})
		case "assistant":
			c.sendUpdate(sessionID, acp.SessionUpdate{
				SessionUpdateKind: acp.UpdateAgentMessageChunk,
				Content:           textContent(*msg.Content + "\n"),
			})
		}
	}
}

func (c *acpConn) prompt(req acp.PromptRequest) (any, *acp.RPCError) {
	active := c.manager.Get(req.SessionID)
	if active == nil {
		return nil, invalidParams(fmt.Errorf("unknown sessionId %q (create one with session/new)", req.SessionID))
	}

	var text strings.Builder
	for _, block := range req.Prompt {
		if block.Type == "text" {
			text.WriteString(block.Text)
		}
	}
	if text.Len() == 0 {
		return nil, invalidParams(fmt.Errorf("prompt contains no text content"))
	}

	events := &acpTurnEvents{conn: c, sessionID: req.SessionID}
	if err := active.RunTurn(c.ctx, text.String(), events); err != nil {
		// ACP allows one prompt per session at a time; the manager rejects
		// rather than queues a concurrent one, so the client gets an
		// immediate, explicit answer.
		if errors.Is(err, agent.ErrSessionBusy) {
			return nil, invalidParams(fmt.Errorf("session %q is already processing a prompt", req.SessionID))
		}
		return nil, internalError(err)
	}

	return acp.PromptResponse{StopReason: "end_turn"}, nil
}

func (c *acpConn) sendUpdate(sessionID string, update acp.SessionUpdate) {
	notif := acp.SessionUpdateNotification{SessionID: sessionID, Update: update}
	if err := c.peer.Notify(acp.MethodSessionUpdate, notif); err != nil {
		klog.FromContext(c.ctx).Error(err, "failed to send session/update")
	}
}

// acpTurnEvents streams agent.RunTurn progress to the ACP client, and
// forwards tool call approval to it via session/request_permission.
type acpTurnEvents struct {
	conn      *acpConn
	sessionID string
}

func (e *acpTurnEvents) AssistantMessage(_ context.Context, text string) {
	e.conn.sendUpdate(e.sessionID, acp.SessionUpdate{
		SessionUpdateKind: acp.UpdateAgentMessageChunk,
		Content:           textContent(text),
	})
}

func (e *acpTurnEvents) ApproveToolCall(ctx context.Context, call llm.ToolCall) (bool, error) {
	params := acp.RequestPermissionParams{
		SessionID: e.sessionID,
		ToolCall:  toolCallUpdate(call, "pending"),
		Options: []acp.PermissionOption{
			{OptionID: "allow", Name: "Allow", Kind: acp.PermissionAllowOnce},
			{OptionID: "reject", Name: "Reject", Kind: acp.PermissionRejectOnce},
		},
	}

	var result acp.RequestPermissionResult
	if err := e.conn.peer.Call(ctx, acp.MethodRequestPermission, params, &result); err != nil {
		return false, err
	}
	return result.Outcome.Outcome == acp.PermissionSelected && result.Outcome.OptionID == "allow", nil
}

func (e *acpTurnEvents) ToolCallStarted(_ context.Context, call llm.ToolCall) {
	e.conn.sendUpdate(e.sessionID, toolCallUpdate(call, "in_progress"))
}

func (e *acpTurnEvents) ToolCallFinished(_ context.Context, call llm.ToolCall, err error) {
	status := "completed"
	if err != nil {
		status = "failed"
	}
	update := toolCallUpdate(call, status)
	update.SessionUpdateKind = acp.UpdateToolCallUpdate
	e.conn.sendUpdate(e.sessionID, update)
}

// toolCallUpdate renders an LLM tool call as an ACP tool_call update.
func toolCallUpdate(call llm.ToolCall, status string) acp.SessionUpdate {
	update := acp.SessionUpdate{
		SessionUpdateKind: acp.UpdateToolCall,
		ToolCallID:        call.ID,
		Title:             fmt.Sprintf("%s(%s)", call.Function.Name, call.Function.Arguments),
		ToolKind:          toolKind(call.Function.Name),
		Status:            status,
	}
	if json.Valid([]byte(call.Function.Arguments)) {
		update.RawInput = json.RawMessage(call.Function.Arguments)
	}
	return update
}

// toolKind maps the tool names of the basic and geminicli toolsets onto the
// ACP tool kind vocabulary.
func toolKind(toolName string) string {
	switch toolName {
	case "run_command", geminicli.ShellToolName:
		return "execute"
	case "ls", toolbox.ListDirectoryToolName, toolbox.GlobToolName, toolbox.GrepToolName:
		return "search"
	case "read", toolbox.ReadFileToolName, toolbox.ReadManyFilesToolName:
		return "read"
	case "write", toolbox.WriteFileToolName, toolbox.ReplaceToolName:
		return "edit"
	default:
		return "other"
	}
}

func textContent(text string) json.RawMessage {
	data, err := json.Marshal(acp.ContentBlock{Type: "text", Text: text})
	if err != nil {
		// A ContentBlock of strings cannot fail to marshal.
		panic(err)
	}
	return data
}

func invalidParams(err error) *acp.RPCError {
	return &acp.RPCError{Code: acp.InvalidParams, Message: err.Error()}
}

func internalError(err error) *acp.RPCError {
	return &acp.RPCError{Code: acp.InternalError, Message: err.Error()}
}
