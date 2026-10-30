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
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	processv1 "sigs.k8s.io/agent-sandbox/packages/sandboxd/spec/process/v1"
)

const testToken = "s3cret-token"

func digestOf(token string) []byte {
	d := sha256.Sum256([]byte(token))
	return d[:]
}

func newAuthServer(t *testing.T, token string) (*Server, string) {
	t.Helper()
	root := t.TempDir()
	var digest []byte
	if token != "" {
		digest = digestOf(token)
	}
	srv, err := New(Options{RootDir: root, AuthTokenSHA256: digest, Log: logr.Discard()})
	require.NoError(t, err)
	return srv, root
}

func newSignedAuthServer(t *testing.T, pub ed25519.PublicKey, audience string) (*Server, string) {
	t.Helper()
	root := t.TempDir()
	srv, err := New(Options{
		RootDir:       root,
		AuthPublicKey: pub,
		AuthAudience:  audience,
		Log:           logr.Discard(),
	})
	require.NoError(t, err)
	return srv, root
}

func newAuthGRPCClient(t *testing.T, srv *Server) processv1.ProcessServiceClient {
	t.Helper()
	lis := bufconn.Listen(1024 * 1024)
	grpcServer := grpc.NewServer(srv.GRPCServerOptions()...)
	srv.RegisterGRPC(grpcServer)
	go func() { _ = grpcServer.Serve(lis) }()
	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return lis.DialContext(ctx)
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() {
		_ = conn.Close()
		grpcServer.Stop()
	})
	return processv1.NewProcessServiceClient(conn)
}

func withAuth(ctx context.Context, value string) context.Context {
	return metadata.AppendToOutgoingContext(ctx, "authorization", value)
}

func TestRESTRequiresBearerToken(t *testing.T) {
	srv, root := newAuthServer(t, testToken)
	handler := srv.RESTHandler()
	put := func(auth string) int {
		h := http.Header{}
		if auth != "" {
			h.Set("Authorization", auth)
		}
		return doRequest(handler, http.MethodPut, "/v1/files/a.txt", []byte("x"), h).Code
	}

	for _, auth := range []string{"", "Bearer wrong", "Basic " + testToken, testToken, "Bearer"} {
		require.Equal(t, http.StatusUnauthorized, put(auth), "auth %q", auth)
	}
	_, err := os.Stat(filepath.Join(root, "a.txt"))
	require.True(t, os.IsNotExist(err), "a refused PUT must not write")

	rec := doRequest(handler, http.MethodGet, "/v1/files/a.txt", nil, nil)
	require.Equal(t, http.StatusUnauthorized, rec.Code)
	require.Equal(t, "UNAUTHENTICATED", decodeError(t, rec).Code)
	require.Contains(t, rec.Header().Get("WWW-Authenticate"), "Bearer")

	require.Equal(t, http.StatusNoContent, put("Bearer "+testToken))
	require.Equal(t, http.StatusNoContent, put("bearer "+testToken))
}

func TestRESTHealthOpenWithoutToken(t *testing.T) {
	srv, _ := newAuthServer(t, testToken)
	rec := doRequest(srv.RESTHandler(), http.MethodGet, "/v1/health", nil, nil)
	require.Equal(t, http.StatusOK, rec.Code)

	rec = doRequest(srv.RESTHandler(), http.MethodGet, "/v1/metadata", nil, nil)
	require.Equal(t, http.StatusUnauthorized, rec.Code)
}

func TestGRPCRequiresBearerToken(t *testing.T) {
	srv, _ := newAuthServer(t, testToken)
	client := newAuthGRPCClient(t, srv)
	req := &processv1.ExecuteRequest{Config: &processv1.ProcessConfig{Command: []string{"true"}}}

	for _, ctx := range []context.Context{
		testCtx(t),
		withAuth(testCtx(t), "Bearer wrong"),
		withAuth(testCtx(t), testToken),
	} {
		_, err := client.Execute(ctx, req)
		require.Equal(t, codes.Unauthenticated, status.Code(err))
	}

	stream, err := client.Start(testCtx(t), &processv1.StartRequest{Config: req.GetConfig()})
	require.NoError(t, err)
	_, err = stream.Recv()
	require.Equal(t, codes.Unauthenticated, status.Code(err))

	resp, err := client.Execute(withAuth(testCtx(t), "Bearer "+testToken), req)
	require.NoError(t, err)
	require.Equal(t, int32(0), resp.GetExitCode())

	stream, err = client.Start(withAuth(testCtx(t), "Bearer "+testToken),
		&processv1.StartRequest{Config: req.GetConfig()})
	require.NoError(t, err)
	for {
		ev, err := stream.Recv()
		require.NoError(t, err)
		if ev.GetExit() != nil {
			require.Equal(t, int32(0), ev.GetExit().GetExitCode())
			break
		}
	}
}

func TestNoTokenLeavesAPIOpen(t *testing.T) {
	srv, _ := newAuthServer(t, "")
	require.Empty(t, srv.GRPCServerOptions())
	rec := doRequest(srv.RESTHandler(), http.MethodPut, "/v1/files/a.txt", []byte("x"), nil)
	require.Equal(t, http.StatusNoContent, rec.Code)
}

func TestParseAuthTokenSHA256(t *testing.T) {
	sum := sha256.Sum256([]byte(testToken))
	got, err := ParseAuthTokenSHA256(hex.EncodeToString(sum[:]) + "\n")
	require.NoError(t, err)
	require.Equal(t, sum[:], got)

	for _, bad := range []string{"", "zz", hex.EncodeToString(sum[:16])} {
		_, err := ParseAuthTokenSHA256(bad)
		require.Error(t, err, "digest %q", bad)
	}
}

func TestNewRejectsShortDigest(t *testing.T) {
	_, err := New(Options{RootDir: t.TempDir(), AuthTokenSHA256: []byte("short"), Log: logr.Discard()})
	require.Error(t, err)
}

func TestLoadAuthTokenFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "token")
	require.NoError(t, os.WriteFile(path, []byte(testToken+"\n"), 0o600))
	got, err := LoadAuthTokenFile(path)
	require.NoError(t, err)
	require.Equal(t, digestOf(testToken), got)

	require.NoError(t, os.WriteFile(path, []byte(" \n"), 0o600))
	_, err = LoadAuthTokenFile(path)
	require.Error(t, err)

	_, err = LoadAuthTokenFile(filepath.Join(dir, "missing"))
	require.Error(t, err)
}

func TestSignedRESTRequiresBearerToken(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)
	srv, root := newSignedAuthServer(t, pub, "test-aud")
	handler := srv.RESTHandler()

	token, err := SignToken(priv, "test-aud", 5*time.Minute)
	require.NoError(t, err)

	put := func(auth string) int {
		h := http.Header{}
		if auth != "" {
			h.Set("Authorization", auth)
		}
		return doRequest(handler, http.MethodPut, "/v1/files/a.txt", []byte("x"), h).Code
	}

	for _, auth := range []string{"", "Bearer wrong", "Basic " + token, token, "Bearer", "Bearer v1.bad"} {
		require.Equal(t, http.StatusUnauthorized, put(auth), "auth %q", auth)
	}
	_, err = os.Stat(filepath.Join(root, "a.txt"))
	require.True(t, os.IsNotExist(err), "a refused PUT must not write")

	rec := doRequest(handler, http.MethodGet, "/v1/files/a.txt", nil, nil)
	require.Equal(t, http.StatusUnauthorized, rec.Code)
	require.Equal(t, "UNAUTHENTICATED", decodeError(t, rec).Code)
	require.Contains(t, rec.Header().Get("WWW-Authenticate"), "Bearer")

	require.Equal(t, http.StatusNoContent, put("Bearer "+token))
	require.Equal(t, http.StatusNoContent, put("bearer "+token))
}

func TestSignedRESTHealthOpenWithoutToken(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)
	srv, _ := newSignedAuthServer(t, pub, "")
	rec := doRequest(srv.RESTHandler(), http.MethodGet, "/v1/health", nil, nil)
	require.Equal(t, http.StatusOK, rec.Code)

	rec = doRequest(srv.RESTHandler(), http.MethodGet, "/v1/metadata", nil, nil)
	require.Equal(t, http.StatusUnauthorized, rec.Code)
}

func TestSignedGRPCRequiresBearerToken(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)
	srv, _ := newSignedAuthServer(t, pub, "test-aud")
	client := newAuthGRPCClient(t, srv)
	req := &processv1.ExecuteRequest{Config: &processv1.ProcessConfig{Command: []string{"true"}}}

	token, err := SignToken(priv, "test-aud", 5*time.Minute)
	require.NoError(t, err)

	for _, ctx := range []context.Context{
		testCtx(t),
		withAuth(testCtx(t), "Bearer wrong"),
		withAuth(testCtx(t), token),
	} {
		_, err := client.Execute(ctx, req)
		require.Equal(t, codes.Unauthenticated, status.Code(err))
	}

	stream, err := client.Start(testCtx(t), &processv1.StartRequest{Config: req.GetConfig()})
	require.NoError(t, err)
	_, err = stream.Recv()
	require.Equal(t, codes.Unauthenticated, status.Code(err))

	resp, err := client.Execute(withAuth(testCtx(t), "Bearer "+token), req)
	require.NoError(t, err)
	require.Equal(t, int32(0), resp.GetExitCode())

	stream, err = client.Start(withAuth(testCtx(t), "Bearer "+token),
		&processv1.StartRequest{Config: req.GetConfig()})
	require.NoError(t, err)
	for {
		ev, err := stream.Recv()
		require.NoError(t, err)
		if ev.GetExit() != nil {
			require.Equal(t, int32(0), ev.GetExit().GetExitCode())
			break
		}
	}
}

func TestSignedTokenWrongKey(t *testing.T) {
	pub1, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)
	_, priv2, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	srv, _ := newSignedAuthServer(t, pub1, "")
	handler := srv.RESTHandler()
	client := newAuthGRPCClient(t, srv)

	token, err := SignToken(priv2, "", 5*time.Minute)
	require.NoError(t, err)

	h := http.Header{}
	h.Set("Authorization", "Bearer "+token)
	rec := doRequest(handler, http.MethodGet, "/v1/files/a.txt", nil, h)
	require.Equal(t, http.StatusUnauthorized, rec.Code)

	req := &processv1.ExecuteRequest{Config: &processv1.ProcessConfig{Command: []string{"true"}}}
	_, err = client.Execute(withAuth(testCtx(t), "Bearer "+token), req)
	require.Equal(t, codes.Unauthenticated, status.Code(err))
}

func TestSignedTokenExpired(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	srv, _ := newSignedAuthServer(t, pub, "")
	handler := srv.RESTHandler()
	client := newAuthGRPCClient(t, srv)

	// Token with TTL in the past is already expired.
	token, err := SignToken(priv, "", -time.Minute)
	require.NoError(t, err)

	h := http.Header{}
	h.Set("Authorization", "Bearer "+token)
	rec := doRequest(handler, http.MethodGet, "/v1/files/a.txt", nil, h)
	require.Equal(t, http.StatusUnauthorized, rec.Code)

	req := &processv1.ExecuteRequest{Config: &processv1.ProcessConfig{Command: []string{"true"}}}
	_, err = client.Execute(withAuth(testCtx(t), "Bearer "+token), req)
	require.Equal(t, codes.Unauthenticated, status.Code(err))
}

func TestSignedTokenFutureIat(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	srv, _ := newSignedAuthServer(t, pub, "")
	handler := srv.RESTHandler()
	client := newAuthGRPCClient(t, srv)

	now := time.Now()

	// Token with iat 61 seconds into future (exceeds 60s clock skew) must be rejected.
	futureToken, err := SignTokenAt(priv, "", 5*time.Minute, now.Add(65*time.Second))
	require.NoError(t, err)

	h := http.Header{}
	h.Set("Authorization", "Bearer "+futureToken)
	rec := doRequest(handler, http.MethodGet, "/v1/files/a.txt", nil, h)
	require.Equal(t, http.StatusUnauthorized, rec.Code)

	req := &processv1.ExecuteRequest{Config: &processv1.ProcessConfig{Command: []string{"true"}}}
	_, err = client.Execute(withAuth(testCtx(t), "Bearer "+futureToken), req)
	require.Equal(t, codes.Unauthenticated, status.Code(err))

	// Token with iat 30 seconds into future (within 60s clock skew) must be accepted.
	skewToken, err := SignTokenAt(priv, "", 5*time.Minute, now.Add(30*time.Second))
	require.NoError(t, err)

	h.Set("Authorization", "Bearer "+skewToken)
	rec = doRequest(handler, http.MethodPut, "/v1/files/skew.txt", []byte("x"), h)
	require.Equal(t, http.StatusNoContent, rec.Code)

	resp, err := client.Execute(withAuth(testCtx(t), "Bearer "+skewToken), req)
	require.NoError(t, err)
	require.Equal(t, int32(0), resp.GetExitCode())
}

func TestSignedTokenAudience(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	// Server with audience required.
	srv, _ := newSignedAuthServer(t, pub, "ns/sandbox-1")
	handler := srv.RESTHandler()

	correctToken, err := SignToken(priv, "ns/sandbox-1", 5*time.Minute)
	require.NoError(t, err)
	wrongToken, err := SignToken(priv, "ns/sandbox-2", 5*time.Minute)
	require.NoError(t, err)
	emptyAudToken, err := SignToken(priv, "", 5*time.Minute)
	require.NoError(t, err)

	h := http.Header{}
	h.Set("Authorization", "Bearer "+wrongToken)
	rec := doRequest(handler, http.MethodGet, "/v1/files/a.txt", nil, h)
	require.Equal(t, http.StatusUnauthorized, rec.Code)

	h.Set("Authorization", "Bearer "+emptyAudToken)
	rec = doRequest(handler, http.MethodGet, "/v1/files/a.txt", nil, h)
	require.Equal(t, http.StatusUnauthorized, rec.Code)

	h.Set("Authorization", "Bearer "+correctToken)
	rec = doRequest(handler, http.MethodPut, "/v1/files/a.txt", []byte("x"), h)
	require.Equal(t, http.StatusNoContent, rec.Code)

	// Server without audience required accepts any audience.
	openSrv, _ := newSignedAuthServer(t, pub, "")
	openHandler := openSrv.RESTHandler()

	h.Set("Authorization", "Bearer "+wrongToken)
	rec = doRequest(openHandler, http.MethodPut, "/v1/files/b.txt", []byte("x"), h)
	require.Equal(t, http.StatusNoContent, rec.Code)

	h.Set("Authorization", "Bearer "+emptyAudToken)
	rec = doRequest(openHandler, http.MethodPut, "/v1/files/c.txt", []byte("x"), h)
	require.Equal(t, http.StatusNoContent, rec.Code)
}

func TestSignedTokenMalformed(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	srv, _ := newSignedAuthServer(t, pub, "")
	handler := srv.RESTHandler()

	validToken, err := SignToken(priv, "", 5*time.Minute)
	require.NoError(t, err)
	parts := strings.Split(validToken, ".")
	require.Len(t, parts, 3)

	malformedTokens := []string{
		"v1",
		"v1.payload",
		"v1.payload.sig.extra",
		"v2." + parts[1] + "." + parts[2],
		"v1.not-base64-url!.sig",
		"v1." + parts[1] + ".not-base64-url!",
		"v1." + base64.RawURLEncoding.EncodeToString([]byte("not json")) + "." + parts[2],
		"v1." + parts[1] + "." + base64.RawURLEncoding.EncodeToString([]byte("short-sig")),
	}

	for _, token := range malformedTokens {
		h := http.Header{}
		h.Set("Authorization", "Bearer "+token)
		rec := doRequest(handler, http.MethodGet, "/v1/files/a.txt", nil, h)
		require.Equal(t, http.StatusUnauthorized, rec.Code, "token %q", token)
	}
}

func TestServerNewAuthValidation(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)
	digest := digestOf("test")

	// Combining AuthTokenSHA256 and AuthPublicKey must be rejected.
	_, err = New(Options{
		RootDir:         t.TempDir(),
		AuthTokenSHA256: digest,
		AuthPublicKey:   pub,
		Log:             logr.Discard(),
	})
	require.Error(t, err)

	// AuthAudience without AuthPublicKey must be rejected.
	_, err = New(Options{
		RootDir:      t.TempDir(),
		AuthAudience: "aud",
		Log:          logr.Discard(),
	})
	require.Error(t, err)

	// A negative AuthMaxTokenTTL must be rejected.
	_, err = New(Options{
		RootDir:         t.TempDir(),
		AuthPublicKey:   pub,
		AuthMaxTokenTTL: -time.Second,
		Log:             logr.Discard(),
	})
	require.Error(t, err)

	// AuthPublicKey with invalid length must be rejected.
	_, err = New(Options{
		RootDir:       t.TempDir(),
		AuthPublicKey: []byte("short"),
		Log:           logr.Discard(),
	})
	require.Error(t, err)
}

func TestParseAuthPublicKey(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	// Standard base64, padded, with surrounding whitespace.
	got, err := ParseAuthPublicKey(base64.StdEncoding.EncodeToString(pub) + "\n")
	require.NoError(t, err)
	require.Equal(t, pub, got)

	// Standard base64, unpadded.
	got, err = ParseAuthPublicKey(base64.RawStdEncoding.EncodeToString(pub))
	require.NoError(t, err)
	require.Equal(t, pub, got)

	// The URL-safe alphabet is not accepted: find a key whose standard
	// encoding uses '+' or '/' so the two alphabets differ, then check the
	// URL-safe form is rejected rather than decoded as a different key.
	for {
		k, _, err := ed25519.GenerateKey(rand.Reader)
		require.NoError(t, err)
		std := base64.StdEncoding.EncodeToString(k)
		if !strings.ContainsAny(std, "+/") {
			continue
		}
		_, err = ParseAuthPublicKey(base64.URLEncoding.EncodeToString(k))
		require.Error(t, err)
		break
	}

	// Empty string
	_, err = ParseAuthPublicKey("   \n")
	require.Error(t, err)

	// Invalid base64
	_, err = ParseAuthPublicKey("invalid!!")
	require.Error(t, err)

	// Wrong length (16 bytes)
	_, err = ParseAuthPublicKey(base64.StdEncoding.EncodeToString(make([]byte, 16)))
	require.Error(t, err)
}

func TestLoadAuthPublicKeyFile(t *testing.T) {
	pub, _, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	dir := t.TempDir()
	path := filepath.Join(dir, "pubkey")
	require.NoError(t, os.WriteFile(path, []byte(base64.StdEncoding.EncodeToString(pub)+"\n"), 0o600))

	got, err := LoadAuthPublicKeyFile(path)
	require.NoError(t, err)
	require.Equal(t, pub, got)

	// A raw binary key is not accepted; the file must hold base64.
	rawPath := filepath.Join(dir, "pubkey.bin")
	require.NoError(t, os.WriteFile(rawPath, pub, 0o600))
	_, err = LoadAuthPublicKeyFile(rawPath)
	require.Error(t, err)

	// Empty file
	emptyPath := filepath.Join(dir, "empty")
	require.NoError(t, os.WriteFile(emptyPath, []byte("  \n"), 0o600))
	_, err = LoadAuthPublicKeyFile(emptyPath)
	require.Error(t, err)

	// Missing file
	_, err = LoadAuthPublicKeyFile(filepath.Join(dir, "missing"))
	require.Error(t, err)
}

func TestSignedTokenLifetimeCapped(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)
	req := &processv1.ExecuteRequest{Config: &processv1.ProcessConfig{Command: []string{"true"}}}

	// Default cap is one hour: a two-hour token is rejected even though it is
	// correctly signed and not yet expired.
	srv, _ := newSignedAuthServer(t, pub, "")
	handler := srv.RESTHandler()
	client := newAuthGRPCClient(t, srv)

	long, err := SignToken(priv, "", 2*time.Hour)
	require.NoError(t, err)
	h := http.Header{}
	h.Set("Authorization", "Bearer "+long)
	require.Equal(t, http.StatusUnauthorized, doRequest(handler, http.MethodGet, "/v1/files/a.txt", nil, h).Code)
	_, err = client.Execute(withAuth(testCtx(t), "Bearer "+long), req)
	require.Equal(t, codes.Unauthenticated, status.Code(err))

	short, err := SignToken(priv, "", 5*time.Minute)
	require.NoError(t, err)
	h.Set("Authorization", "Bearer "+short)
	require.Equal(t, http.StatusNoContent, doRequest(handler, http.MethodPut, "/v1/files/a.txt", []byte("x"), h).Code)

	// A token whose exp is not after its iat is rejected.
	inverted, err := SignTokenAt(priv, "", -time.Minute, time.Now().Add(2*time.Minute))
	require.NoError(t, err)
	h.Set("Authorization", "Bearer "+inverted)
	require.Equal(t, http.StatusUnauthorized, doRequest(handler, http.MethodGet, "/v1/files/a.txt", nil, h).Code)

	// Raising the cap admits the two-hour token.
	relaxed, err := New(Options{
		RootDir:         t.TempDir(),
		AuthPublicKey:   pub,
		AuthMaxTokenTTL: 3 * time.Hour,
		Log:             logr.Discard(),
	})
	require.NoError(t, err)
	h.Set("Authorization", "Bearer "+long)
	require.Equal(t, http.StatusNoContent, doRequest(relaxed.RESTHandler(), http.MethodPut, "/v1/files/b.txt", []byte("x"), h).Code)
}

func TestSignToken(t *testing.T) {
	pub, priv, err := ed25519.GenerateKey(rand.Reader)
	require.NoError(t, err)

	token, err := SignToken(priv, "aud1", 5*time.Minute)
	require.NoError(t, err)

	parts := strings.Split(token, ".")
	require.Len(t, parts, 3)
	require.Equal(t, "v1", parts[0])

	payloadBytes, err := base64.RawURLEncoding.DecodeString(parts[1])
	require.NoError(t, err)
	sigBytes, err := base64.RawURLEncoding.DecodeString(parts[2])
	require.NoError(t, err)

	var payload SignedTokenPayload
	require.NoError(t, json.Unmarshal(payloadBytes, &payload))
	require.Equal(t, "aud1", payload.Aud)
	require.NotEmpty(t, payload.Nonce)
	require.Greater(t, payload.Exp, payload.Iat)
	require.Equal(t, int64(300), payload.Exp-payload.Iat)

	require.True(t, ed25519.Verify(pub, payloadBytes, sigBytes))

	// Invalid private key size
	_, err = SignToken([]byte("short"), "aud1", 5*time.Minute)
	require.Error(t, err)
}
