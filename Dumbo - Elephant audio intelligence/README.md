# 🐘 ElephantVoices Rumble Denoiser

**HackSMU VII — ElephantVoices Challenge**

AI-powered tool to remove mechanical noise (airplanes, cars, generators) from elephant recordings, isolating low-frequency rumble vocalizations for bioacoustic research.

## How It Works

1. **Load** audio recording (WAV) and compute spectrogram (STFT)
2. **Classify** noise type using **Google Gemini Vision API** — sends spectrogram image for AI analysis
3. **Denoise** using noise-type-specific strategy:
   - **Generator**: Spectral subtraction + harmonic masking (exploits steady tonal nature)
   - **Airplane**: Wiener filter + spectral subtraction (adapts to sweeping broadband noise)
   - **Car**: Harmonic masking + gentle subtraction (preserves overlapping low frequencies)
4. **Reconstruct** cleaned audio via inverse STFT
5. **Narrate** results using **ElevenLabs TTS**
6. **Serve** via Auth0-secured Flask API on **DigitalOcean**

## Quick Start

```bash
pip install -r requirements.txt

# Set API keys
export GEMINI_API_KEY="your-gemini-key"
export ELEVENLABS_API_KEY="your-elevenlabs-key"

# Single file
python elephant_denoiser.py -i recording.wav -o cleaned.wav

# With known noise type (skip Gemini)
python elephant_denoiser.py -i recording.wav --noise-type generator

# With call timing from spreadsheet
python elephant_denoiser.py -i recording.wav --call-start 2.5 --call-end 8.1

# Batch — all 44 data
python elephant_denoiser.py -i ./data/ --batch --spreadsheet calls.csv
```

## Sponsor Integrations

### 🟡 Gemini API (Google)
Spectrogram images are sent to Gemini 2.0 Flash's vision capability to:
- Classify noise type (airplane / car / generator)
- Describe elephant call patterns visible in the spectrogram
- Recommend optimal denoising strategy

Get your key: https://aistudio.google.com/apikey

### 🔵 ElevenLabs
Generates voice narration summarizing analysis results for each processed recording — noise type detected, denoising strategy used, and call descriptions.

Get your key: https://elevenlabs.io (use the hackathon promo code from MLH)

### 🔒 Auth0
The Flask API (`api_server.py`) uses Auth0 JWT verification to secure endpoints:
- `POST /denoise` — upload WAV, get cleaned audio back
- `POST /analyze` — classify noise type only

Setup:
1. Create free Auth0 account → Applications → Create API
2. Set `AUTH0_DOMAIN` and `AUTH0_AUDIENCE` env vars
3. Use Auth0 dashboard to create test tokens

### 🌊 DigitalOcean
Deploy the API on a DigitalOcean Droplet:

```bash
# On a fresh Ubuntu Droplet ($200 free credits)
sudo apt update && sudo apt install -y python3-pip python3-venv libsndfile1
git clone <your-repo> && cd elephant-denoiser
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export AUTH0_DOMAIN="your-tenant.auth0.com"
export AUTH0_AUDIENCE="https://elephant-denoiser-api"
export GEMINI_API_KEY="..."
export ELEVENLABS_API_KEY="..."

python api_server.py
```

## Technical Details

**Why harmonic masking works:** Elephant rumbles have a strict harmonic structure — frequencies at integer multiples of the fundamental (10-20 Hz). Mechanical noise lacks this structure. By detecting the harmonic series and creating a frequency mask, we preserve elephant vocalizations while suppressing non-harmonic noise energy.

**Key parameters** (in `Config` class):
| Parameter | Default | Description |
|-----------|---------|-------------|
| NFFT | 4096 | FFT window — large for good low-freq resolution |
| FUNDAMENTAL_LOW/HIGH | 8-25 Hz | Expected elephant fundamental range |
| MAX_HARMONIC_FREQ | 1200 Hz | Highest harmonic to preserve |
| NOISE_REDUCTION_STRENGTH | 0.8 | Aggressiveness of spectral subtraction |

## Team

Built at HackSMU VII for ElephantVoices.
# Elephant-Voice-Detector
