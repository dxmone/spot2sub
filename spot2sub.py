#!/usr/bin/env python3
"""
Spot2Sub

CLI tool to copy a Spotify playlist into Navidrome (Subsonic API):
- Prompts for Spotify Client ID/Secret and Redirect URI
- Authenticates and lists user playlists for selection
- Fetches tracks from the selected playlist
- Prompts for Navidrome URL and credentials
- Searches library on Navidrome and creates a mirrored playlist
- Reports missing/unmatched tracks at the end

Notes:
- You must register the Redirect URI in your Spotify App settings.
- Required Spotify scopes: playlist-read-private, playlist-read-collaborative
- No external dependencies besides the standard library and requests.
"""

from __future__ import annotations

import base64
import argparse
import dataclasses
import getpass
import hashlib
import http.server
import json
import os
import queue
import random
import re
import socket
import string
import sys
import threading
import time
import urllib.parse
import webbrowser
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import requests
except ImportError:  # pragma: no cover
    print("This script requires the 'requests' package. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)


SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"


def prompt(msg: str, default: Optional[str] = None, secret: bool = False) -> str:
    if secret:
        val = getpass.getpass(f"{msg}{' ['+default+']' if default else ''}: ") if not os.getenv("SPOT2SUB_NONINTERACTIVE") else ""
    else:
        val = input(f"{msg}{' ['+default+']' if default else ''}: ") if not os.getenv("SPOT2SUB_NONINTERACTIVE") else ""
    val = val.strip()
    if not val and default is not None:
        return default
    return val


def gen_state(n: int = 16) -> str:
    return ''.join(random.choices(string.ascii_letters + string.digits, k=n))


class OAuthCallbackHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # silence noise
        return

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        self.server.last_query = params  # type: ignore[attr-defined]
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html><body><h3>Auth complete. You can close this window.</h3></body></html>")


def start_local_server(host: str, port: int) -> Tuple[http.server.HTTPServer, threading.Thread]:
    httpd = http.server.HTTPServer((host, port), OAuthCallbackHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, t


def wait_for_code(httpd: http.server.HTTPServer, timeout: int = 120) -> Dict[str, List[str]]:
    start = time.time()
    while time.time() - start < timeout:
        params = getattr(httpd, 'last_query', None)
        if params:
            return params
        time.sleep(0.1)
    raise TimeoutError("Timed out waiting for OAuth redirect")


def spotify_authorize(client_id: str, client_secret: str, redirect_uri: str, scopes: List[str]) -> Dict[str, str]:
    """Perform Spotify Authorization Code flow with a local redirect server if possible.

    Returns an access token dict with access_token, refresh_token, expires_in, token_type.
    """
    parsed = urllib.parse.urlparse(redirect_uri)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Redirect URI must be http or https")

    # Prepare authorize URL
    state = gen_state()
    auth_params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "state": state,
        "show_dialog": "false",
    }
    url = f"{SPOTIFY_AUTH_URL}?{urllib.parse.urlencode(auth_params)}"
    print("Open the following URL to authorize Spotify (a browser window should open):\n", url)

    # Try to run a local server to capture the redirect
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    httpd = None
    server_thread = None
    try:
        if host in ("localhost", "127.0.0.1") and port not in (80, 443):
            httpd, server_thread = start_local_server(host, port)
            webbrowser.open(url)
            params = wait_for_code(httpd)
            if params.get("state", [None])[0] != state:
                raise RuntimeError("State mismatch in OAuth response")
            code = params.get("code", [None])[0]
        else:
            # If we can't bind a local server or using 80/443, fall back to manual paste
            webbrowser.open(url)
            redirected = prompt("Paste the full redirected URL here")
            p = urllib.parse.urlparse(redirected)
            params = urllib.parse.parse_qs(p.query)
            if params.get("state", [None])[0] != state:
                raise RuntimeError("State mismatch in OAuth response")
            code = params.get("code", [None])[0]
    finally:
        if httpd is not None:
            httpd.shutdown()
        if server_thread is not None:
            server_thread.join(timeout=1)

    if not code:
        raise RuntimeError("Authorization code not found")

    # Exchange code for token
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    resp = requests.post(SPOTIFY_TOKEN_URL, headers=headers, data=data, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Token exchange failed: {resp.status_code} {resp.text}")
    token = resp.json()
    return token


def spotify_refresh_token(client_id: str, client_secret: str, refresh_token: str) -> Dict[str, str]:
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    headers = {"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
    resp = requests.post(SPOTIFY_TOKEN_URL, headers=headers, data=data, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Token refresh failed: {resp.status_code} {resp.text}")
    token = resp.json()
    if "refresh_token" not in token:
        token["refresh_token"] = refresh_token
    return token


def sp_get(url: str, access_token: str, params: Optional[dict] = None) -> dict:
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.get(url, headers=headers, params=params or {}, timeout=30)
    if resp.status_code == 401:
        raise PermissionError("Spotify token expired or unauthorized")
    resp.raise_for_status()
    return resp.json()


def fetch_all_spotify_playlists(access_token: str) -> List[dict]:
    playlists = []
    url = f"{SPOTIFY_API_BASE}/me/playlists"
    params = {"limit": 50}
    while url:
        data = sp_get(url, access_token, params)
        playlists.extend(data.get("items", []))
        url = data.get("next")
        params = None  # Spotify next already includes query
    return playlists


def fetch_spotify_playlist_tracks(access_token: str, playlist_id: str) -> List[dict]:
    tracks: List[dict] = []
    url = f"{SPOTIFY_API_BASE}/playlists/{playlist_id}/tracks"
    params = {"limit": 100}
    while url:
        data = sp_get(url, access_token, params)
        items = data.get("items", [])
        for it in items:
            t = it.get("track")
            if not t:
                continue
            if t.get("is_local"):
                # Skip local-only tracks (no Spotify ID)
                continue
            tracks.append(t)
        url = data.get("next")
        params = None
    return tracks


# ---------------- Navidrome / Subsonic client ----------------

@dataclasses.dataclass
class SubsonicClient:
    base_url: str
    username: str
    password: str
    client: str = "spot2sub"
    api_version: str = "1.16.1"
    use_token_auth: bool = True

    def _auth_params(self) -> dict:
        if self.use_token_auth:
            salt = gen_state(12)
            token = hashlib.md5((self.password + salt).encode()).hexdigest()
            return {
                "u": self.username,
                "t": token,
                "s": salt,
            }
        else:
            return {
                "u": self.username,
                "p": self.password,
            }

    def _base_params(self) -> dict:
        return {
            **self._auth_params(),
            "v": self.api_version,
            "c": self.client,
            "f": "json",
        }

    def _endpoint(self, name: str) -> str:
        base = self.base_url.rstrip('/')
        # Navidrome follows Subsonic .view endpoints
        if not name.endswith('.view'):
            name = name + '.view'
        return f"{base}/rest/{name}"

    def ping(self) -> dict:
        r = requests.get(self._endpoint('ping'), params=self._base_params(), timeout=30)
        r.raise_for_status()
        return r.json()

    def search_songs(self, query: str, count: int = 5) -> List[dict]:
        params = {**self._base_params(), "query": query, "songCount": count}
        r = requests.get(self._endpoint('search3'), params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        songs = data.get('subsonic-response', {}).get('searchResult3', {}).get('song', [])
        return songs if isinstance(songs, list) else ([] if songs is None else [songs])

    def create_playlist(self, name: str, song_ids: List[str]) -> dict:
        params = self._base_params()
        form = [("name", name)] + [("songId", sid) for sid in song_ids]
        r = requests.post(self._endpoint('createPlaylist'), params=params, data=form, timeout=60)
        r.raise_for_status()
        return r.json()


# ---------------- Config handling ----------------

DEFAULT_CONFIG_TEMPLATE = {
    "spotify": {
        "client_id": "YOUR_SPOTIFY_CLIENT_ID",
        "client_secret": "YOUR_SPOTIFY_CLIENT_SECRET",
        "redirect_uri": "http://127.0.0.1:8888/callback",
    },
    "navidrome": {
        "base_url": "http://localhost:4533",
        "username": "navidrome",
        "password": "changeme",
        "client_name": "spot2sub",
    },
}


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        print(f"Config file not found: {path}", file=sys.stderr)
        print("Create this file before running. Example config:\n")
        print(json.dumps(DEFAULT_CONFIG_TEMPLATE, indent=2))
        sys.exit(2)

    try:
        with open(path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"Failed to read config file {path}: {e}", file=sys.stderr)
        sys.exit(2)

    # Validate required fields
    missing = []
    for section, keys in (("spotify", ["client_id", "client_secret", "redirect_uri"]),
                          ("navidrome", ["base_url", "username", "password"])):
        if section not in cfg or not isinstance(cfg[section], dict):
            missing.append(section)
            continue
        for k in keys:
            if not cfg[section].get(k):
                missing.append(f"{section}.{k}")

    if missing:
        print("Config is missing required fields:")
        for m in missing:
            print(f" - {m}")
        print(f"Edit {path} and rerun.")
        sys.exit(2)

    return cfg


# ---------------- Matching helpers ----------------

_PUNCT_RE = re.compile(r"[\-_,.;:!\?\(\)\[\]\{\}\"\'\u2019\u2018\u201C\u201D]")


def normalize_text(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    s = s.strip()
    return s


def best_song_match(candidates: List[dict], track_name: str, artist: str, album: Optional[str], duration_ms: Optional[int]) -> Optional[dict]:
    if not candidates:
        return None
    tn = normalize_text(track_name)
    ar = normalize_text(artist)
    al = normalize_text(album) if album else None
    best = None
    best_score = -1e9
    for c in candidates:
        c_title = normalize_text(c.get('title', ''))
        c_artist = normalize_text(c.get('artist', ''))
        c_album = normalize_text(c.get('album', ''))
        c_dur = int(c.get('duration', 0)) * 1000 if c.get('duration') is not None else None

        score = 0
        if tn == c_title:
            score += 100
        elif tn in c_title or c_title in tn:
            score += 60
        else:
            # fuzzy-ish token overlap
            tn_tokens = set(tn.split())
            c_tokens = set(c_title.split())
            overlap = len(tn_tokens & c_tokens)
            score += overlap * 8

        if ar == c_artist:
            score += 80
        elif ar in c_artist or c_artist in ar:
            score += 40

        if al and al == c_album:
            score += 25
        elif al and (al in c_album or c_album in al):
            score += 10

        if duration_ms and c_dur:
            delta = abs(duration_ms - c_dur)
            if delta < 2000:
                score += 30
            elif delta < 5000:
                score += 10
            else:
                score -= min(30, delta / 1000)

        if score > best_score:
            best_score = score
            best = c

    # Basic threshold to avoid absurd mismatches
    if best_score < 40:
        return None
    return best


def select_from_list(items: List[str], title: str) -> int:
    print(f"\n{title}")
    for i, name in enumerate(items):
        print(f"  [{i+1:3d}] {name}")
    while True:
        choice = input("Enter number: ").strip()
        if not choice.isdigit():
            print("Please enter a number.")
            continue
        idx = int(choice) - 1
        if 0 <= idx < len(items):
            return idx
        print("Out of range, try again.")


def main():
    print("=== Spot2Sub: Copy a Spotify playlist into Navidrome ===\n")
    # Config
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", dest="config", default=os.environ.get("SPOT2SUB_CONFIG", "config.json"))
    parser.add_argument("-h", "--help", action="store_true")
    args, _ = parser.parse_known_args()
    if args.help:
        print("Usage: python spot2sub.py [--config path]")
        print("Requires a config file. See config.json template.")
        return 0

    cfg = load_config(args.config)

    # Spotify creds
    client_id = cfg["spotify"]["client_id"]
    client_secret = cfg["spotify"]["client_secret"]
    redirect_uri = cfg["spotify"].get("redirect_uri", "http://127.0.0.1:8888/callback")

    print("\nAuthenticating with Spotify (hardcoded client + redirect)...")
    scopes = ["playlist-read-private", "playlist-read-collaborative"]
    try:
        token = spotify_authorize(client_id, client_secret, redirect_uri, scopes)
    except Exception as e:
        print(f"Spotify auth failed: {e}", file=sys.stderr)
        return 1

    access_token = token["access_token"]
    refresh_token = token.get("refresh_token")
    expires_at = time.time() + int(token.get("expires_in", 3600))

    def ensure_token():
        nonlocal access_token, refresh_token, expires_at
        if time.time() >= expires_at - 30 and refresh_token:
            t = spotify_refresh_token(client_id, client_secret, refresh_token)
            access_token = t["access_token"]
            refresh_token = t.get("refresh_token", refresh_token)
            expires_at = time.time() + int(t.get("expires_in", 3600))

    # Fetch playlists
    print("Fetching Spotify playlists...")
    ensure_token()
    try:
        playlists = fetch_all_spotify_playlists(access_token)
    except Exception as e:
        print(f"Failed to retrieve playlists: {e}", file=sys.stderr)
        return 1

    if not playlists:
        print("No playlists found on this Spotify account.")
        return 0

    names = [f"{p['name']} ({p['tracks']['total']} tracks)" for p in playlists]
    idx = select_from_list(names, "Select a Spotify playlist:")
    playlist = playlists[idx]
    playlist_name = playlist["name"]
    playlist_id = playlist["id"]

    print(f"\nFetching tracks for '{playlist_name}'...")
    ensure_token()
    try:
        tracks = fetch_spotify_playlist_tracks(access_token, playlist_id)
    except Exception as e:
        print(f"Failed to retrieve playlist tracks: {e}", file=sys.stderr)
        return 1

    print(f"Found {len(tracks)} tracks.\n")

    # Navidrome creds
    base_url = cfg["navidrome"]["base_url"]
    username = cfg["navidrome"]["username"]
    password = cfg["navidrome"]["password"]
    client_name = cfg["navidrome"].get("client_name", "spot2sub")

    sub = SubsonicClient(base_url=base_url, username=username, password=password, client=client_name)
    print("\nPinging Navidrome...")
    try:
        pong = sub.ping()
        status = pong.get('subsonic-response', {}).get('status')
        if status != 'ok':
            raise RuntimeError(f"Ping status: {status}")
    except Exception as e:
        print(f"Failed to connect to Navidrome: {e}", file=sys.stderr)
        return 1

    print("Searching Navidrome for matching tracks and building playlist...")
    matched_ids: List[str] = []
    missing: List[Dict[str, str]] = []

    for i, t in enumerate(tracks, 1):
        name = t.get('name', '')
        artists = ", ".join(a.get('name', '') for a in (t.get('artists') or []))
        album = (t.get('album') or {}).get('name')
        duration_ms = t.get('duration_ms')
        query = f"{name} {artists}"

        try:
            candidates = sub.search_songs(query, count=8)
        except Exception as e:
            print(f"  [{i}/{len(tracks)}] Search failed for '{name}' - {e}")
            missing.append({"track": name, "artist": artists, "album": album or "", "reason": "search error"})
            continue

        best = best_song_match(candidates, name, artists, album, duration_ms)
        if best:
            matched_ids.append(str(best.get('id')))
            print(f"  [{i}/{len(tracks)}] ✓ {artists} - {name}")
        else:
            print(f"  [{i}/{len(tracks)}] ✗ {artists} - {name} (not found)")
            missing.append({"track": name, "artist": artists, "album": album or "", "reason": "no match"})

    if matched_ids:
        nav_playlist_name = f"{playlist_name} (from Spotify)"
        print(f"\nCreating Navidrome playlist '{nav_playlist_name}' with {len(matched_ids)} tracks...")
        try:
            sub.create_playlist(nav_playlist_name, matched_ids)
            print("Playlist created successfully.")
        except Exception as e:
            print(f"Failed to create playlist: {e}", file=sys.stderr)
    else:
        print("\nNo tracks matched; skipping playlist creation.")

    # Report missing
    print("\n=== Missing/Unmatched Tracks ===")
    if not missing:
        print("All tracks were matched in Navidrome. Nice!")
    else:
        for m in missing:
            print(f"- {m['artist']} - {m['track']} ({m.get('album','')}) [{m.get('reason','')}]")
        # Save to file
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", playlist_name)[:80]
        out_path = f"missing_{safe_name}.txt"
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(f"Missing tracks for playlist: {playlist_name}\n\n")
                for m in missing:
                    f.write(f"- {m['artist']} - {m['track']} ({m.get('album','')}) [{m.get('reason','')}]\n")
            print(f"\nSaved missing list to {out_path}")
        except Exception as e:
            print(f"Failed to write missing report: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
