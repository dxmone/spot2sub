# Spot2Sub

Copy a Spotify playlist into Navidrome (Subsonic API) while reporting any tracks that are missing from your Navidrome library.

## What It Does
- Authenticates with Spotify, lists your playlists, and lets you select one.
- Fetches all tracks in the selected playlist.
- Connects to your Navidrome server via the Subsonic API.
- Searches for each Spotify track in your Navidrome library and builds a matching playlist.
- Writes a `missing_<playlist>.txt` report for tracks that could not be matched.

## Requirements
- Python 3.8+
- `requests` library: `pip install requests`
- A Spotify Developer app with the Redirect URI set to `http://127.0.0.1:8888/callback`.
- Access to your Navidrome instance (URL, username, password).

## Setup
1. Create a Spotify app (https://developer.spotify.com/dashboard) and in your app settings add the Redirect URI:
   - `http://127.0.0.1:8888/callback`
2. Create a `config.json` in the project directory with your credentials. Example:
   ```json
   {
     "spotify": {
       "client_id": "YOUR_SPOTIFY_CLIENT_ID",
       "client_secret": "YOUR_SPOTIFY_CLIENT_SECRET",
       "redirect_uri": "http://127.0.0.1:8888/callback"
     },
     "navidrome": {
       "base_url": "http://localhost:4533",
       "username": "navidrome",
       "password": "changeme",
       "client_name": "spot2sub"
     }
   }
   ```
   Notes:
   - The `redirect_uri` must exactly match one of the Redirect URIs configured for your Spotify app.
   - `client_name` is the identifier used when calling the Subsonic API.
3. Ensure `config.json` is not committed. A `.gitignore` entry is already included to ignore `config.json`.

## Usage
- Install dependencies:
  - `pip install requests`
- Run the script:
  - `python spot2sub.py`
  - Optional: `python spot2sub.py --config /path/to/config.json`
- A browser window opens for Spotify authorization. After approving, the script proceeds automatically.
- Select the playlist to mirror when prompted.
- The script creates a new Navidrome playlist named `<Spotify name> (from Spotify)` with all matched tracks.
- A `missing_<playlist>.txt` file is written listing any tracks not found in your library.

## How Matching Works
- Uses Navidrome’s `search3` endpoint to find candidate songs by title and artist.
- Applies a heuristic to score candidates by title, artist, album, and duration similarity.
- Skips local-only Spotify tracks (no Spotify ID) since they can’t be retrieved via the API.

## Troubleshooting
- INVALID_CLIENT: Invalid redirect URI
  - Ensure the Redirect URI in `config.json` exactly matches the one registered in your Spotify app: `http://127.0.0.1:8888/callback`.
- Browser didn’t open or port in use
  - The script starts a local server on `127.0.0.1:8888`. Make sure port 8888 is free, or change the `redirect_uri` in `config.json` and your Spotify app settings to use a different port.
- Spotify 401 Unauthorized
  - Credentials may be wrong or expired. Re-run and re-authorize.
- Navidrome connection issues
  - Check `base_url`, username, and password. For HTTPS with self-signed certs, consider configuring your system certs or a reverse proxy.
- Many tracks marked missing
  - Ensure your library metadata (titles/artists/albums) are consistent. Adjust metadata or try more complete search queries in Navidrome.

## Notes
- The script uses Subsonic token authentication (md5(password+salt)); your password is never sent in plaintext when `use_token_auth` is enabled.
- The created playlist uses the order of matched tracks as found. Unmatched tracks are reported but not added.

## License
This repository does not include a license by default. Add one if you intend to share or distribute.
