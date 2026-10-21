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
	"errors"
	"fmt"
	"strings"

	"k8s.io/klog/v2"

	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/llm"
	"sigs.k8s.io/agent-sandbox/examples/sandboxed-tools/pkg/tools"
)

// GhToolName is the name of the gh tool.
const GhToolName = "gh"

// maxGhOutputSize caps the output of one gh invocation; issue and PR
// listings can be large, and the supervisor only needs enough to act on.
const maxGhOutputSize = 256 * 1024

// GhTool runs the GitHub CLI in the supervisor's sandbox. It is the
// supervisor's view of GitHub (issues, pull requests, comments, checks); the
// engineering work itself happens in child sessions.
//
// The arguments are passed to gh directly, without a shell, so the model
// cannot run anything but gh with this tool. gh authenticates with the
// GH_TOKEN environment variable of the sandbox.
type GhTool struct {
	// Args are the gh arguments, e.g. ["issue", "view", "42", "--repo", "org/repo"].
	Args []string `json:"args"`
}

func (t *GhTool) Schema() llm.Tool {
	return llm.Tool{
		Type: "function",
		Function: llm.ToolFunction{
			Name: GhToolName,
			Description: "Runs the GitHub CLI (`gh`) with the given arguments, for example [\"issue\", \"view\", \"42\", \"--repo\", \"org/repo\"] or [\"pr\", \"list\", \"--repo\", \"org/repo\", \"--json\", \"number,title,url\"]. " +
				"Use it to read issues and pull requests, comment on them, and check CI status. Prefer --json output when you need to act on the result. " +
				"Returns the combined output and, if non-zero, the exit code.",
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"args": map[string]any{
						"type":        "array",
						"items":       map[string]any{"type": "string"},
						"description": "The arguments to pass to gh, one per element, without the leading \"gh\".",
					},
				},
				"required": []string{"args"},
			},
		},
	}
}

func (t *GhTool) Run(ctx context.Context, sandbox tools.Sandbox) (llm.Message, error) {
	if len(t.Args) == 0 {
		return llm.Message{}, errors.New("args is required")
	}
	// Don't log the arguments: they can carry issue text or comments.
	klog.FromContext(ctx).Info("running gh in sandbox", "subcommand", t.Args[0])

	combined := tools.NewLimitedWriter(maxGhOutputSize)
	res, err := sandbox.ExecCommand(ctx, tools.ExecCommandOptions{
		Command: append([]string{"gh"}, t.Args...),
		Stdout:  combined,
		Stderr:  combined,
	})
	if err != nil {
		return llm.Message{}, err
	}

	output := combined.String()
	if strings.TrimSpace(output) == "" {
		output = "(empty)"
	}
	if combined.Truncated() {
		output += fmt.Sprintf("\n[Output truncated at %d bytes]", combined.Len())
	}
	content := "Output: " + output
	if res.ExitCode != 0 {
		content += fmt.Sprintf("\nExit Code: %d", res.ExitCode)
	}
	return llm.Message{Content: &content}, nil
}
