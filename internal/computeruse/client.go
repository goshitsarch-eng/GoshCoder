package computeruse

// Stdio MCP client for the computer-use-linux server. The binary speaks the
// rmcp 2024-11-05 protocol: newline-delimited JSON-RPC 2.0 over
// stdin/stdout, capability discovery via tools/list. Desktop input is
// stateful, so calls are serialized on one session; the tool additionally
// registers as sequential so the agent never runs it concurrently with
// other tools.

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"sync"
	"time"
)

const (
	mcpProtocolVersion = "2024-11-05"
	initTimeout        = 15 * time.Second
	callTimeout        = 120 * time.Second
	// maxLineBytes bounds one server response line; screenshots travel as
	// base64 image payloads, which the server caps at 2 MiB of image bytes
	// before encoding.
	maxLineBytes = 64 << 20
)

// ToolInfo is one tools/list entry.
type ToolInfo struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"inputSchema,omitempty"`
	Annotations *struct {
		ReadOnlyHint    *bool `json:"readOnlyHint,omitempty"`
		DestructiveHint *bool `json:"destructiveHint,omitempty"`
		IdempotentHint  *bool `json:"idempotentHint,omitempty"`
		OpenWorldHint   *bool `json:"openWorldHint,omitempty"`
	} `json:"annotations,omitempty"`
}

// ContentItem is one tools/call result content item. Screenshots arrive as
// type "image" with base64 data and a mime type.
type ContentItem struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Data     string `json:"data,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
}

// CallResult is a tools/call result.
type CallResult struct {
	Content []ContentItem `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
	ID      *int64 `json:"id,omitempty"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *int64          `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Session is one spawned MCP server process. It is created lazily on first
// use and restarted transparently when the process dies.
type Session struct {
	binaryPath string
	// argv is the argument tail, normally just "mcp"; tests re-exec the test
	// binary with extra flags in front.
	argv []string

	mu      sync.Mutex
	cmd     *exec.Cmd
	stdin   io.WriteCloser
	stdout  *bufio.Reader
	nextID  int64
	tools   []ToolInfo
	started bool
}

// NewSession prepares a session for the binary; nothing is spawned yet.
func NewSession(binaryPath string) *Session {
	return &Session{binaryPath: binaryPath, argv: []string{"mcp"}}
}

// start spawns and initializes the server. Callers hold s.mu.
func (s *Session) start(ctx context.Context) error {
	if s.started {
		return nil
	}
	// The child inherits the environment, so the server's own variables
	// (COMPUTER_USE_LINUX_COSMIC_HELPER, COMPUTER_USE_LINUX_ENABLE_SHELL,
	// backend forcing, DBUS_SESSION_BUS_ADDRESS, ...) keep working.
	cmd := exec.Command(s.binaryPath, s.argv...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		stdin.Close()
		return err
	}
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		stdin.Close()
		return fmt.Errorf("start %s: %w", s.binaryPath, err)
	}
	s.cmd = cmd
	s.stdin = stdin
	s.stdout = bufio.NewReaderSize(stdout, 1<<20)
	s.nextID = 0
	s.started = true

	initCtx, cancel := context.WithTimeout(ctx, initTimeout)
	defer cancel()
	result, err := s.roundTripLocked(initCtx, "initialize", map[string]any{
		"protocolVersion": mcpProtocolVersion,
		"capabilities":    map[string]any{},
		"clientInfo":      map[string]any{"name": "goshcoder", "version": "1"},
	})
	if err != nil {
		s.stopLocked()
		return fmt.Errorf("initialize %s: %w", ServerName, err)
	}
	var initialized struct {
		ProtocolVersion string `json:"protocolVersion"`
	}
	if json.Unmarshal(result, &initialized) != nil || initialized.ProtocolVersion == "" {
		s.stopLocked()
		return fmt.Errorf("initialize %s: unexpected result %s", ServerName, string(result))
	}
	if err := s.writeLocked(rpcRequest{JSONRPC: "2.0", Method: "notifications/initialized"}); err != nil {
		s.stopLocked()
		return err
	}
	return nil
}

func (s *Session) writeLocked(request rpcRequest) error {
	encoded, err := json.Marshal(request)
	if err != nil {
		return err
	}
	encoded = append(encoded, '\n')
	_, err = s.stdin.Write(encoded)
	return err
}

// roundTripLocked sends one request and reads until its response arrives,
// skipping server-initiated notifications. Callers hold s.mu.
func (s *Session) roundTripLocked(ctx context.Context, method string, params any) (json.RawMessage, error) {
	s.nextID++
	id := s.nextID
	if err := s.writeLocked(rpcRequest{JSONRPC: "2.0", Method: method, Params: params, ID: &id}); err != nil {
		return nil, err
	}

	type lineResult struct {
		response rpcResponse
		err      error
	}
	results := make(chan lineResult, 1)
	go func() {
		for {
			line, err := readLine(s.stdout)
			if err != nil {
				results <- lineResult{err: err}
				return
			}
			if len(line) == 0 {
				continue
			}
			var response rpcResponse
			if json.Unmarshal(line, &response) != nil {
				continue
			}
			// Notifications and unrelated ids are skipped; the session
			// serializes requests, so the next matching id is ours.
			if response.ID == nil || *response.ID != id {
				continue
			}
			results <- lineResult{response: response}
			return
		}
	}()

	select {
	case <-ctx.Done():
		// The reader goroutine is stuck on a dead or wedged server; kill the
		// process so the pipe closes and the goroutine exits.
		s.stopLocked()
		return nil, ctx.Err()
	case result := <-results:
		if result.err != nil {
			s.stopLocked()
			return nil, fmt.Errorf("%s exited: %w", ServerName, result.err)
		}
		if result.response.Error != nil {
			return nil, fmt.Errorf("%s: %s (code %d)", ServerName, result.response.Error.Message, result.response.Error.Code)
		}
		return result.response.Result, nil
	}
}

func readLine(reader *bufio.Reader) ([]byte, error) {
	var line []byte
	for {
		chunk, isPrefix, err := reader.ReadLine()
		if err != nil {
			return nil, err
		}
		line = append(line, chunk...)
		if len(line) > maxLineBytes {
			return nil, errors.New("MCP response line exceeds the size limit")
		}
		if !isPrefix {
			return line, nil
		}
	}
}

func (s *Session) stopLocked() {
	if s.cmd != nil {
		if s.stdin != nil {
			s.stdin.Close()
		}
		_ = s.cmd.Process.Kill()
		_ = s.cmd.Wait()
	}
	s.cmd, s.stdin, s.stdout = nil, nil, nil
	s.started = false
	s.tools = nil
}

// Close terminates the server process.
func (s *Session) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stopLocked()
}

// Tools lists the server's tools, spawning and initializing it on first use.
// The list is cached for the life of the process.
func (s *Session) Tools(ctx context.Context) ([]ToolInfo, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.start(ctx); err != nil {
		return nil, err
	}
	if s.tools != nil {
		return s.tools, nil
	}
	callCtx, cancel := context.WithTimeout(ctx, initTimeout)
	defer cancel()
	result, err := s.roundTripLocked(callCtx, "tools/list", nil)
	if err != nil {
		return nil, err
	}
	var decoded struct {
		Tools []ToolInfo `json:"tools"`
	}
	if err := json.Unmarshal(result, &decoded); err != nil {
		return nil, fmt.Errorf("decode tools/list: %w", err)
	}
	s.tools = decoded.Tools
	return s.tools, nil
}

// Call executes one tool. Calls are serialized: desktop input is stateful
// and the server documents avoiding concurrent tool calls.
func (s *Session) Call(ctx context.Context, name string, args map[string]any) (CallResult, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err := s.start(ctx); err != nil {
		return CallResult{}, err
	}
	if args == nil {
		args = map[string]any{}
	}
	callCtx, cancel := context.WithTimeout(ctx, callTimeout)
	defer cancel()
	result, err := s.roundTripLocked(callCtx, "tools/call", map[string]any{"name": name, "arguments": args})
	if err != nil {
		return CallResult{}, err
	}
	var decoded CallResult
	if err := json.Unmarshal(result, &decoded); err != nil {
		return CallResult{}, fmt.Errorf("decode tools/call: %w", err)
	}
	return decoded, nil
}
