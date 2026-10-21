#!/usr/bin/env python3
import sys
import socket
import subprocess
import hashlib
import base64
import threading

PORT = 3000
COMMAND = "gemini --acp"

def handle_client(conn):
    client_addr = conn.getpeername() if hasattr(conn, 'getpeername') else "Unknown"
    print(f"New connection from {client_addr}", file=sys.stderr)
    proc = None
    try:
        # 1. Perform WebSocket Handshake
        data = conn.recv(4096).decode('utf-8', errors='ignore')
        if not data:
            print(f"[{client_addr}] Connection closed by client before handshake.", file=sys.stderr)
            conn.close()
            return

        lines = data.split('\r\n')
        if lines:
            print(f"[{client_addr}] Request: {lines[0]}", file=sys.stderr)

        headers = {}
        for line in lines[1:]:
            if ': ' in line:
                k, v = line.split(': ', 1)
                headers[k.lower()] = v.strip()
        
        print(f"[{client_addr}] Headers received: {headers}", file=sys.stderr)
        
        key = headers.get('sec-websocket-key')
        if not key:
            print(f"[{client_addr}] Handshake failed: Sec-WebSocket-Key header missing in request headers: {list(headers.keys())}", file=sys.stderr)
            conn.close()
            return
            
        accept_val = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode('utf-8')).digest()
        ).decode('utf-8')
        
        protocol_header = ""
        if 'sec-websocket-protocol' in headers:
            proto = headers['sec-websocket-protocol']
            protocol_header = f"Sec-WebSocket-Protocol: {proto}\r\n"
            print(f"[{client_addr}] Client requested protocol: {proto}", file=sys.stderr)

        response = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"{protocol_header}"
            f"Sec-WebSocket-Accept: {accept_val}\r\n\r\n"
        )
        conn.sendall(response.encode('utf-8'))
        print(f"[{client_addr}] Handshake successful. Upgraded to WebSocket. Protocol: {headers.get('sec-websocket-protocol', 'None')}", file=sys.stderr)
        
        # 2. Spawn the Gemini ACP Agent Process
        print(f"[{client_addr}] Spawning process: '{COMMAND}'", file=sys.stderr)
        try:
            proc = subprocess.Popen(
                COMMAND, shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=sys.stderr,
            )
        except Exception as e:
            print(f"[{client_addr}] Failed to spawn process '{COMMAND}': {e}", file=sys.stderr)
            conn.close()
            return
        
        # 3. Thread to read agent stdout -> send as WebSocket frames
        def stdout_to_ws():
            try:
                while True:
                    line = proc.stdout.readline()
                    if not line:
                        exit_code = proc.poll()
                        print(f"[{client_addr}] Agent process stdout EOF. Exit code: {exit_code}", file=sys.stderr)
                        break
                    
                    line_str = line.decode('utf-8', errors='ignore').strip()
                    log_line = line_str #if len(line_str) <= 500 else line_str[:500] + f" ... [truncated {len(line_str)-500} chars]"
                    print(f"[{client_addr}] << AGENT to CLIENT: {log_line}", file=sys.stderr)

                    length = len(line)
                    header = bytearray([0x81]) # FIN + Text opcode
                    if length <= 125:
                        header.append(length)
                    elif length <= 65535:
                        header.append(126)
                        header.extend(length.to_bytes(2, 'big'))
                    else:
                        header.append(127)
                        header.extend(length.to_bytes(8, 'big'))
                    conn.sendall(header + line)
            except Exception as e:
                print(f"[{client_addr}] Exception in stdout_to_ws: {e}", file=sys.stderr)
            finally:
                print(f"[{client_addr}] Closing connection and terminating process...", file=sys.stderr)
                conn.close()
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
                except Exception as pe:
                    print(f"[{client_addr}] Error terminating process: {pe}", file=sys.stderr)

        threading.Thread(target=stdout_to_ws, daemon=True).start()
        
        # 4. Read WebSocket frames -> write to agent stdin
        while True:
            head = conn.recv(2)
            if not head or len(head) < 2:
                print(f"[{client_addr}] Connection closed or EOF when reading frame header.", file=sys.stderr)
                break
            mask_len = head[1]
            masked = bool(mask_len & 0x80)
            length = mask_len & 0x7f
            
            if length == 126:
                len_bytes = conn.recv(2)
                if len(len_bytes) < 2:
                    print(f"[{client_addr}] Failed to read 2 bytes for extended frame length.", file=sys.stderr)
                    break
                length = int.from_bytes(len_bytes, 'big')
            elif length == 127:
                len_bytes = conn.recv(8)
                if len(len_bytes) < 8:
                    print(f"[{client_addr}] Failed to read 8 bytes for extended frame length.", file=sys.stderr)
                    break
                length = int.from_bytes(len_bytes, 'big')
                
            mask_key = None
            if masked:
                mask_key = conn.recv(4)
                if len(mask_key) < 4:
                    print(f"[{client_addr}] Failed to read 4 bytes for mask key.", file=sys.stderr)
                    break
            
            payload = bytearray()
            while len(payload) < length:
                chunk = conn.recv(length - len(payload))
                if not chunk:
                    print(f"[{client_addr}] Connection closed while reading frame payload.", file=sys.stderr)
                    break
                payload.extend(chunk)
                
            if len(payload) < length:
                break
                
            if masked and mask_key:
                for i in range(len(payload)):
                    payload[i] ^= mask_key[i % 4]
            
            msg_str = payload.decode('utf-8', errors='ignore').strip()
            log_msg = msg_str if len(msg_str) <= 500 else msg_str[:500] + f" ... [truncated {len(msg_str)-500} chars]"
            print(f"[{client_addr}] >> CLIENT to AGENT: {log_msg}", file=sys.stderr)
                    
            proc.stdin.write(payload)
            proc.stdin.flush()
            
    except Exception as e:
        print(f"[{client_addr}] Connection error: {e}", file=sys.stderr)
    finally:
        print(f"[{client_addr}] Cleaning up connection...", file=sys.stderr)
        conn.close()
        if proc:
            try:
                proc.terminate()
            except ProcessLookupError:
                pass
            except Exception as pe:
                print(f"[{client_addr}] Error terminating process in finally: {pe}", file=sys.stderr)

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('0.0.0.0', PORT))
    server.listen(5)
    print(f"ACP Bridge listening on ws://0.0.0.0:{PORT}", file=sys.stderr)
    
    try:
        while True:
            conn, _ = server.accept()
            threading.Thread(target=handle_client, args=(conn,), daemon=True).start()
    except KeyboardInterrupt:
        print("\nShutting down bridge...", file=sys.stderr)
    finally:
        server.close()

if __name__ == "__main__":
    main()
