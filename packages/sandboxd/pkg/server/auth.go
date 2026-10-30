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

package server

import (
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

// healthPath is served without a token so the kubelet's readiness probe,
// which carries no per-sandbox credential, keeps working. Every other path,
// /v1/metadata included, requires the token.
const healthPath = "/v1/health"

// AuthTokenDigestSize is the length in bytes of the token digest
// Options.AuthTokenSHA256 holds.
const AuthTokenDigestSize = sha256.Size

// ParseAuthTokenSHA256 decodes the hex SHA-256 digest of the bearer token
// (64 hex characters). Configuring the digest rather than the token keeps the
// token itself out of the pod spec, the process environment and anything
// copied from them.
func ParseAuthTokenSHA256(hexDigest string) ([]byte, error) {
	digest, err := hex.DecodeString(strings.TrimSpace(hexDigest))
	if err != nil {
		return nil, fmt.Errorf("auth token sha256: %w", err)
	}
	if len(digest) != AuthTokenDigestSize {
		return nil, fmt.Errorf("auth token sha256: want %d bytes, got %d", AuthTokenDigestSize, len(digest))
	}
	return digest, nil
}

// LoadAuthTokenFile reads the bearer token from path and returns its SHA-256
// digest. Surrounding whitespace (e.g. a trailing newline) is ignored; an
// empty token is an error, so a misconfigured mount never silently disables
// authentication.
func LoadAuthTokenFile(path string) ([]byte, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read auth token file: %w", err)
	}
	token := strings.TrimSpace(string(raw))
	if token == "" {
		return nil, errors.New("auth token file is empty")
	}
	digest := sha256.Sum256([]byte(token))
	return digest[:], nil
}

// DefaultAuthMaxTokenTTL is the longest lifetime (exp - iat) a signed token
// may claim when Options.AuthMaxTokenTTL is zero. A cap enforced by the daemon
// bounds how long a captured token stays usable regardless of how the client
// mints tokens.
const DefaultAuthMaxTokenTTL = time.Hour

// ParseAuthPublicKey decodes a raw 32-byte ed25519 public key encoded with the
// standard base64 alphabet (RFC 4648 section 4), with or without padding. No
// other encoding is accepted, so a misconfigured key fails at startup rather
// than being read as a different key.
func ParseAuthPublicKey(base64Key string) (ed25519.PublicKey, error) {
	s := strings.TrimSpace(base64Key)
	if s == "" {
		return nil, errors.New("auth public key is empty")
	}
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		b, err = base64.RawStdEncoding.DecodeString(s)
	}
	if err != nil {
		return nil, fmt.Errorf("decode auth public key (standard base64): %w", err)
	}
	if len(b) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("auth public key: want %d bytes, got %d", ed25519.PublicKeySize, len(b))
	}
	return ed25519.PublicKey(b), nil
}

// LoadAuthPublicKeyFile reads a standard-base64 ed25519 public key from path,
// in the same format ParseAuthPublicKey accepts. Surrounding whitespace is
// ignored; an empty file is an error.
func LoadAuthPublicKeyFile(path string) (ed25519.PublicKey, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read auth public key file: %w", err)
	}
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" {
		return nil, errors.New("auth public key file is empty")
	}
	return ParseAuthPublicKey(trimmed)
}

// authChecker validates "Authorization: Bearer <token>" headers.
type authChecker interface {
	valid(header string) bool
}

// tokenChecker validates "Authorization: Bearer <token>" values against the
// configured token's SHA-256 digest in constant time. Only the digest is
// held, so the daemon never needs the token itself.
type tokenChecker struct {
	want [sha256.Size]byte
}

func newTokenChecker(digest []byte) *tokenChecker {
	c := &tokenChecker{}
	copy(c.want[:], digest)
	return c
}

func (c *tokenChecker) valid(header string) bool {
	scheme, credential, ok := strings.Cut(header, " ")
	if !ok || !strings.EqualFold(scheme, "Bearer") {
		return false
	}
	got := sha256.Sum256([]byte(strings.TrimSpace(credential)))
	return subtle.ConstantTimeCompare(got[:], c.want[:]) == 1
}

// signedTokenChecker verifies asymmetric ed25519-signed bearer tokens.
type signedTokenChecker struct {
	publicKey ed25519.PublicKey
	audience  string
	maxTTL    time.Duration
	clock     func() time.Time
}

func newSignedTokenChecker(pubKey ed25519.PublicKey, audience string, maxTTL time.Duration) *signedTokenChecker {
	if maxTTL <= 0 {
		maxTTL = DefaultAuthMaxTokenTTL
	}
	return &signedTokenChecker{
		publicKey: pubKey,
		audience:  audience,
		maxTTL:    maxTTL,
		clock:     time.Now,
	}
}

func (c *signedTokenChecker) now() time.Time {
	if c.clock != nil {
		return c.clock()
	}
	return time.Now()
}

func decodeBase64URL(s string) ([]byte, error) {
	b, err := base64.RawURLEncoding.DecodeString(s)
	if err == nil {
		return b, nil
	}
	return base64.URLEncoding.DecodeString(s)
}

func (c *signedTokenChecker) valid(header string) bool {
	scheme, credential, ok := strings.Cut(header, " ")
	if !ok || !strings.EqualFold(scheme, "Bearer") {
		return false
	}
	token := strings.TrimSpace(credential)
	parts := strings.Split(token, ".")
	if len(parts) != 3 || parts[0] != "v1" {
		return false
	}

	payloadBytes, err := decodeBase64URL(parts[1])
	if err != nil {
		return false
	}
	sigBytes, err := decodeBase64URL(parts[2])
	if err != nil {
		return false
	}
	if len(sigBytes) != ed25519.SignatureSize {
		return false
	}
	if !ed25519.Verify(c.publicKey, payloadBytes, sigBytes) {
		return false
	}

	var payload SignedTokenPayload
	if err := json.Unmarshal(payloadBytes, &payload); err != nil {
		return false
	}

	now := c.now().Unix()
	if payload.Exp <= now {
		return false
	}
	if payload.Iat > now+60 {
		return false
	}
	// The daemon, not the client, bounds how long a token can live, so a
	// captured token expires within maxTTL even if the signer asked for more.
	if payload.Exp <= payload.Iat || payload.Exp-payload.Iat > int64(c.maxTTL/time.Second) {
		return false
	}
	if c.audience != "" && payload.Aud != c.audience {
		return false
	}

	return true
}

func authorizeGRPC(ctx context.Context, c authChecker) error {
	md, _ := metadata.FromIncomingContext(ctx)
	if slices.ContainsFunc(md.Get("authorization"), c.valid) {
		return nil
	}
	return status.Error(codes.Unauthenticated, "missing or invalid bearer token")
}

func unaryAuthInterceptor(c authChecker) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, _ *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		if err := authorizeGRPC(ctx, c); err != nil {
			return nil, err
		}
		return handler(ctx, req)
	}
}

func streamAuthInterceptor(c authChecker) grpc.StreamServerInterceptor {
	return func(srv any, ss grpc.ServerStream, _ *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		if err := authorizeGRPC(ss.Context(), c); err != nil {
			return err
		}
		return handler(srv, ss)
	}
}

// authMiddleware rejects REST requests without a valid token, except the health probe.
func authMiddleware(c authChecker, next http.Handler, s *RESTServer) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != healthPath && !c.valid(r.Header.Get("Authorization")) {
			w.Header().Set("WWW-Authenticate", `Bearer realm="sandboxd"`)
			s.writeError(w, http.StatusUnauthorized, "UNAUTHENTICATED",
				"missing or invalid bearer token")
			return
		}
		next.ServeHTTP(w, r)
	})
}
