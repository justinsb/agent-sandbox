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

package acp

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/url"
	"strings"
	"sync"

	"k8s.io/klog/v2"
)

// Limit the max message size to avoid corrupt messages causing us to buffer massive amounts of data
const MAX_MESSAGE_SIZE = 4 * 1024 * 1024 // 4MiB

// The RFC 6455 WebSocket GUID constant used in the opening handshake to calculate Sec-WebSocket-Accept.
const websocketGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

// Agent defines the interface for building custom ACP agents.
type Agent interface {
	// Initialize is called during the initial handshake to request metadata info.
	Initialize(ctx context.Context, clientInfo ClientInfo) (ServerInfo, error)

	// StartSession is invoked when a client requests a new chat session.
	StartSession(ctx context.Context, conn *Connection) (*Session, error)

	// Prompt is invoked when the client submits prompt chunks to the active session.
	Prompt(ctx context.Context, session *Session, prompt string) error
}

// ClientInfo represents standard metadata sent by standard client platforms.
type ClientInfo struct {
	Name    string `json:"name"`
	Title   string `json:"title"`
	Version string `json:"version"`
}

// ServerInfo represents standard metadata identifying the agent.
type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// Session manages the state and notification pipelines of an active communication channel.
type Session struct {
	ID string

	connection *Connection
}

// Notification represents standard JSON-RPC 2.0 notifications.
type Notification struct {
	JsonRpc string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params"`
}

// SessionUpdateParams defines parameters for real-time progress update pushes.
type SessionUpdateParams struct {
	SessionID string        `json:"sessionId"`
	Update    SessionUpdate `json:"update"`
}

// SessionUpdate defines updates containing either thought bubbles or actual message text.
type SessionUpdate struct {
	SessionUpdate string         `json:"sessionUpdate"`
	Content       map[string]any `json:"content,omitempty"`
}

// StreamThought streams reasoning progress chunks back to the client in real-time.
func (s *Session) StreamThought(ctx context.Context, thought string) error {
	notification := Notification{
		JsonRpc: "2.0",
		Method:  "session/update",
		Params: SessionUpdateParams{
			SessionID: s.ID,
			Update: SessionUpdate{
				SessionUpdate: "agent_thought_chunk",
				Content: map[string]any{
					"type": "text",
					"text": thought,
				},
			},
		},
	}

	return s.connection.writeJSONMessage(ctx, notification)
}

// StreamMessage streams response message chunks back to the client in real-time.
func (s *Session) StreamMessage(ctx context.Context, message string) error {
	notification := Notification{
		JsonRpc: "2.0",
		Method:  "session/update",
		Params: SessionUpdateParams{
			SessionID: s.ID,
			Update: SessionUpdate{
				SessionUpdate: "agent_message_chunk",
				Content: map[string]any{
					"type": "text",
					"text": message,
				},
			},
		},
	}
	return s.connection.writeJSONMessage(ctx, notification)
}

// NewUUID generates a dependency-free Version 4 UUID using crypto/rand.
func NewUUID() string {
	uuid := make([]byte, 16)
	_, err := rand.Read(uuid)
	if err != nil {
		panic("failed to generate uuid: " + err.Error())
	}
	// Set version to 4
	uuid[6] = (uuid[6] & 0x0f) | 0x40
	// Set variant to RFC 4122
	uuid[8] = (uuid[8] & 0x3f) | 0x80

	return fmt.Sprintf("%08x-%04x-%04x-%04x-%12x",
		uuid[0:4], uuid[4:6], uuid[6:8], uuid[8:10], uuid[10:16])
}

// JSON-RPC 2.0 Protocol Structs
type Request struct {
	JsonRpc string           `json:"jsonrpc"`
	ID      *json.RawMessage `json:"id,omitempty"`
	Method  string           `json:"method"`
	Params  *json.RawMessage `json:"params,omitempty"`
}

type Response struct {
	JsonRpc string           `json:"jsonrpc"`
	ID      *json.RawMessage `json:"id,omitempty"`
	Result  any              `json:"result,omitempty"`
	Error   *Error           `json:"error,omitempty"`
}

type Error struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type InitializeResult struct {
	ProtocolVersion int            `json:"protocolVersion"`
	Capabilities    map[string]any `json:"capabilities"`
	ServerInfo      ServerInfo     `json:"serverInfo"`
}

// Server implements the high-level WebSocket server managing transport and routing.
type Server struct {
	Agent    Agent
	listener net.Listener

	mu          sync.Mutex
	connections map[string]*Connection
}

// Connection holds the information for an ACP connection
type Connection struct {
	id string

	server *Server
	conn   net.Conn

	mu       sync.Mutex
	sessions map[string]*Session
}

// NewSession is called when the client establishes a new session channel.
func (c *Connection) NewSession(ctx context.Context) (*Session, error) {
	sessionID := NewUUID()

	session := &Session{
		ID:         sessionID,
		connection: c,
	}

	return session, nil
}

func (s *Server) registerConnection(c net.Conn) *Connection {
	id := NewUUID()
	connection := &Connection{
		id:     id,
		server: s,
		conn:   c,
	}
	s.mu.Lock()
	if s.connections == nil {
		s.connections = make(map[string]*Connection)
	}
	s.connections[id] = connection
	s.mu.Unlock()
	return connection
}

func (s *Server) unregisterConnection(c *Connection) {
	s.mu.Lock()
	if s.connections != nil {
		delete(s.connections, c.id)
	}
	s.mu.Unlock()
}

func (s *Server) closeAllConns() {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, conn := range s.connections {
		conn.conn.Close()
	}
	s.connections = nil
}

// ListenAndServe starts the WebSocket server and handles clean shutdown signals.
func (s *Server) ListenAndServe(ctx context.Context, listenAddr string) error {
	log := klog.FromContext(ctx)

	if strings.HasPrefix(listenAddr, "ws://") {
		parsed, err := url.Parse(listenAddr)
		if err == nil {
			listenAddr = parsed.Host
		}
	} else if strings.HasPrefix(listenAddr, "localhost:") || strings.HasPrefix(listenAddr, "locahost:") {
		listenAddr = strings.Replace(listenAddr, "locahost:", "localhost:", 1)
	}

	listener, err := net.Listen("tcp", listenAddr)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", listenAddr, err)
	}
	defer listener.Close()

	s.mu.Lock()
	s.listener = listener
	s.mu.Unlock()

	log.Info("ACP WebSocket server listening", "addr", "ws://"+listener.Addr().String())

	go func() {
		<-ctx.Done()
		s.mu.Lock()
		if s.listener != nil {
			s.listener.Close()
		}
		s.mu.Unlock()
		s.closeAllConns()
	}()

	for {
		conn, err := listener.Accept()
		if err != nil {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				log.Error(err, "Accept error")
				continue
			}
		}

		acpConnection := s.registerConnection(conn)
		go func(c *Connection) {
			c.handleConnection(ctx)
			s.unregisterConnection(c)
		}(acpConnection)
	}
}

func (c *Connection) handleConnection(ctx context.Context) {
	log := klog.FromContext(ctx)

	defer c.conn.Close()
	clientAddr := c.conn.RemoteAddr().String()
	log = log.WithValues("client.addr", clientAddr)

	reader := bufio.NewReader(c.conn)

	// 1. Upgrade connection to WebSocket
	var reqBytes []byte
	for {
		line, err := reader.ReadBytes('\n')
		if err != nil {
			log.Error(err, "Connection closed before handshake")
			return
		}
		reqBytes = append(reqBytes, line...)
		if len(reqBytes) >= 4 && bytes.HasSuffix(reqBytes, []byte("\r\n\r\n")) {
			break
		}
	}

	lines := strings.Split(string(reqBytes), "\r\n")
	if len(lines) == 0 {
		log.Info("Empty request received")
		return
	}
	log.Info("Received HTTP Request", "request", lines[0])

	headers := make(map[string]string)
	for _, line := range lines[1:] {
		if line == "" {
			continue
		}
		parts := strings.SplitN(line, ":", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(strings.ToLower(parts[0]))
			val := strings.TrimSpace(parts[1])
			headers[key] = val
		}
	}

	key, ok := headers["sec-websocket-key"]
	if !ok {
		log.Info("Handshake failed: Sec-WebSocket-Key header missing")
		return
	}

	hash := sha1.New()
	hash.Write([]byte(key + websocketGUID))
	acceptVal := base64.StdEncoding.EncodeToString(hash.Sum(nil))

	protocolHeader := ""
	if proto, ok := headers["sec-websocket-protocol"]; ok {
		protocolHeader = "Sec-WebSocket-Protocol: " + proto + "\r\n"
		log.Info("Client requested protocol", "protocol", proto)
	}

	response := "HTTP/1.1 101 Switching Protocols\r\n" +
		"Upgrade: websocket\r\n" +
		"Connection: Upgrade\r\n" +
		protocolHeader +
		"Sec-WebSocket-Accept: " + acceptVal + "\r\n\r\n"

	_, err := c.conn.Write([]byte(response))
	if err != nil {
		log.Error(err, "Failed to send handshake response")
		return
	}
	log.Info("Handshake successful. Upgraded to WebSocket.")

	// 2. Read WebSocket Frames -> Route and Process JSON-RPC Messages
	for {
		frame, err := c.readWebsocketFrame(ctx)
		if err != nil {
			if errors.Is(err, io.EOF) {
				log.Info("connection closed by client")
			} else {
				log.Error(err, "Error reading frame")
			}
			return
		}

		switch frame.OpCode {
		case 0x01: // Text frame
			msgStr := string(frame.Payload)
			logMsg := msgStr
			if len(logMsg) > 500 {
				logMsg = logMsg[:500] + fmt.Sprintf(" ... [truncated %d chars]", len(logMsg)-500)
			}
			log.Info("Received client message", "message", logMsg)

			reply, err := c.routeMessage(ctx, msgStr)
			if err != nil {
				log.Error(err, "Route error")
				reply = ""
			}

			if reply == nil {
				continue
			}

			if err := c.writeJSONMessage(ctx, reply); err != nil {
				log.Error(err, "Failed to write response")
				return
			}

		case 0x08: // Connection close
			log.Info("Received close frame. Closing connection.")
			_ = c.writeWebsocketFrame(ctx, 0x08, nil)
			return

		case 0x09: // Ping
			log.Info("Received Ping frame. Sending Pong.")
			err = c.writeWebsocketFrame(ctx, 0x0A, frame.Payload)
			if err != nil {
				log.Error(err, "Failed to send Pong reply")
				return
			}
		}
	}
}

func (c *Connection) routeMessage(ctx context.Context, rawMsg string) (any, error) {
	log := klog.FromContext(ctx)

	var req Request
	if err := json.Unmarshal([]byte(rawMsg), &req); err != nil {
		return nil, err
	}

	if req.ID == nil {
		if req.Method == "$/ping" {
			resp := Response{
				JsonRpc: "2.0",
				Result:  "pong",
			}
			respBytes, err := json.Marshal(resp)
			if err != nil {
				return "", err
			}
			return string(respBytes), nil
		}
		log.Info("Received notification. No response required.", "method", req.Method)
		return nil, nil
	}

	resp := Response{
		JsonRpc: "2.0",
		ID:      req.ID,
	}

	switch req.Method {
	case "initialize":
		var params map[string]any
		var clientInfo ClientInfo
		if req.Params != nil {
			_ = json.Unmarshal(*req.Params, &params)
			if infoBytes, err := json.Marshal(params["clientInfo"]); err == nil {
				_ = json.Unmarshal(infoBytes, &clientInfo)
			}
		}

		info, err := c.server.Agent.Initialize(ctx, clientInfo)
		if err != nil {
			return nil, err
		}

		resp.Result = InitializeResult{
			ProtocolVersion: 1,
			Capabilities:    make(map[string]any),
			ServerInfo:      info,
		}

		return resp, nil

	case "session/new":
		session, err := c.server.Agent.StartSession(ctx, c)
		if err != nil {
			return nil, err
		}

		c.mu.Lock()
		if c.sessions == nil {
			c.sessions = make(map[string]*Session)
		}
		c.sessions[session.ID] = session
		c.mu.Unlock()

		resp.Result = map[string]any{
			"sessionId": session.ID,
			"status":    "ok",
		}

		return resp, nil

	case "session/prompt":
		var paramsObj map[string]any
		sessionID := ""
		if req.Params != nil {
			if err := json.Unmarshal(*req.Params, &paramsObj); err == nil {
				if sid, ok := paramsObj["sessionId"].(string); ok {
					sessionID = sid
				}
			}
		}

		c.mu.Lock()
		session, exists := c.sessions[sessionID]
		c.mu.Unlock()

		if !exists {
			resp.Error = &Error{
				Code:    -32602,
				Message: fmt.Sprintf("Invalid session ID: %q", sessionID),
			}
			return resp, nil
		}

		promptText := extractText(req.Params)
		if err := c.server.Agent.Prompt(ctx, session, promptText); err != nil {
			resp.Error = &Error{
				Code:    -32000,
				Message: err.Error(),
			}
			return resp, nil
		}

		resp.Result = map[string]any{
			"stopReason": "end_turn",
		}
		return resp, nil

	case "$/ping":
		resp.Result = "pong"
		return resp, nil

	default:
		resp.Error = &Error{
			Code:    -32601,
			Message: fmt.Sprintf("Method not found: %q", req.Method),
		}

		return resp, nil
	}
}

func extractText(params *json.RawMessage) string {
	if params == nil {
		return ""
	}
	var s string
	if err := json.Unmarshal(*params, &s); err == nil {
		return s
	}
	var obj map[string]any
	if err := json.Unmarshal(*params, &obj); err == nil {
		for _, key := range []string{"text", "content", "message", "prompt", "query"} {
			if val, ok := obj[key]; ok {
				if str, isStr := val.(string); isStr {
					return str
				}
				if list, isList := val.([]any); isList {
					var sb strings.Builder
					for _, item := range list {
						if itemMap, isMap := item.(map[string]any); isMap {
							if textVal, ok := itemMap["text"]; ok {
								if textStr, isTextStr := textVal.(string); isTextStr {
									if sb.Len() > 0 {
										sb.WriteString("\n")
									}
									sb.WriteString(textStr)
								}
							}
						} else if itemStr, isItemStr := item.(string); isItemStr {
							if sb.Len() > 0 {
								sb.WriteString("\n")
							}
							sb.WriteString(itemStr)
						}
					}
					if sb.Len() > 0 {
						return sb.String()
					}
				}
				if itemMap, isMap := val.(map[string]any); isMap {
					if textVal, ok := itemMap["text"]; ok {
						if textStr, isTextStr := textVal.(string); isTextStr {
							return textStr
						}
					}
				}
			}
		}
	}
	return string(*params)
}

type websocketFrame struct {
	OpCode  byte
	Payload []byte
}

func (c *Connection) readWebsocketFrame(ctx context.Context) (*websocketFrame, error) {
	header := make([]byte, 2)
	_, err := io.ReadFull(c.conn, header)
	if err != nil {
		return nil, err
	}

	opcode := header[0] & 0x0F
	masked := (header[1] & 0x80) != 0
	length := uint64(header[1] & 0x7F)

	if length == 126 {
		lenBytes := make([]byte, 2)
		_, err = io.ReadFull(c.conn, lenBytes)
		if err != nil {
			return nil, err
		}
		length = uint64(binary.BigEndian.Uint16(lenBytes))
	} else if length == 127 {
		lenBytes := make([]byte, 8)
		_, err = io.ReadFull(c.conn, lenBytes)
		if err != nil {
			return nil, err
		}
		length = binary.BigEndian.Uint64(lenBytes)
	}

	var maskKey []byte
	if masked {
		maskKey = make([]byte, 4)
		_, err = io.ReadFull(c.conn, maskKey)
		if err != nil {
			return nil, err
		}
	}

	if length >= MAX_MESSAGE_SIZE {
		return nil, fmt.Errorf("payload too large: %d bytes", length)
	}
	payload := make([]byte, length)
	_, err = io.ReadFull(c.conn, payload)
	if err != nil {
		return nil, err
	}

	if masked {
		for i := uint64(0); i < length; i++ {
			payload[i] ^= maskKey[i%4]
		}
	}

	return &websocketFrame{OpCode: opcode, Payload: payload}, nil
}

// writeWebsocketFrame is used to write messages to the client.
// It encodes a websocket frame and sends it over the wire.
func (c *Connection) writeWebsocketFrame(ctx context.Context, opcode byte, payload []byte) error {
	length := uint64(len(payload))

	var frame []byte

	firstByte := 0x80 | (opcode & 0x0F)

	if length <= 125 {
		// Length 0-125: send length as single byte
		frame = make([]byte, 2+len(payload))
		frame[0] = firstByte
		frame[1] = byte(length)
		copy(frame[2:], payload)
	} else if length <= 65535 {
		// Length 126-65535: send 126 followed by 2 byte length
		frame = make([]byte, 2+2+len(payload))
		frame[0] = firstByte
		frame[1] = 126
		binary.BigEndian.PutUint16(frame[2:], uint16(length))
		copy(frame[4:], payload)
	} else {
		// Length > 65535: send 127 followed by 8 byte length
		frame = make([]byte, 2+8+len(payload))
		frame[0] = firstByte
		frame[1] = 127
		binary.BigEndian.PutUint64(frame[2:], length)
		copy(frame[10:], payload)
	}

	_, err := c.conn.Write(frame)
	if err != nil {
		return err
	}
	return nil
}

func (c *Connection) writeJSONMessage(ctx context.Context, v any) error {
	log := klog.FromContext(ctx)

	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("converting message to json: %w", err)
	}

	log.Info("sending message", "message", string(b))
	return c.writeWebsocketFrame(ctx, 0x01, []byte(b))
}
