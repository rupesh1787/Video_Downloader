/**
 * Transcription Service
 * Generates transcript text for supported video URLs.
 */

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const OpenAI = require('openai');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const config = require('../config');
const videoEngine = require('./videoEngine');

class TranscriptionService {
  constructor() {
    this.tempDir = config.tempDir;
    this.ytdlpPath = config.ytdlpPath;
    this.pythonPath = config.pythonPath;
    this.localWhisperEnabled = config.localWhisperEnabled;
    this.whisperModel = config.whisperModel;
    this.openaiClient = config.openaiApiKey
      ? new OpenAI({ apiKey: config.openaiApiKey })
      : null;
    this.googleClient = config.googleAiStudioApiKey
      ? new GoogleGenerativeAI(config.googleAiStudioApiKey)
      : null;
  }

  /**
   * Generate transcript from URL.
   * Tries subtitle extraction first, then optional OpenAI audio transcription.
   * @param {string} url
   * @param {string} jobId
   * @returns {Promise<object>}
   */
  async generateTranscript(url, jobId) {
    const metadata = await videoEngine.getMetadata(url);
    const maxDurationSeconds = config.maxTranscriptionMinutes * 60;

    if ((metadata.duration || 0) > maxDurationSeconds) {
      throw new Error(`Video is too long for transcription (max ${config.maxTranscriptionMinutes} minutes).`);
    }

    const transcriptDir = path.join(this.tempDir, jobId, 'transcript');
    fs.mkdirSync(transcriptDir, { recursive: true });

    let fromCaptions = null;
    try {
      fromCaptions = await this.trySubtitleExtraction(url, transcriptDir);
    } catch (captionError) {
      // Continue to next fallback so one extractor failure does not block transcript generation.
      fromCaptions = null;
    }

    if (fromCaptions?.text) {
      return {
        text: fromCaptions.text,
        source: 'captions',
        language: fromCaptions.language || config.transcriptionLanguage,
        videoTitle: metadata.title,
        durationSeconds: metadata.duration || 0,
        generatedAt: new Date().toISOString()
      };
    }

    const audioPath = await this.downloadAudio(url, transcriptDir);
    try {
      let audioText = '';
      let source = '';
      const providerErrors = [];

      if (this.openaiClient) {
        try {
          const result = await this.withTimeout(this.openaiClient.audio.transcriptions.create({
            file: fs.createReadStream(audioPath),
            model: 'gpt-4o-mini-transcribe',
            language: config.transcriptionLanguage,
            response_format: 'text'
          }), 60000, 'OpenAI transcription timed out.');
          audioText = (result || '').toString().trim();
          source = 'openai-whisper';
        } catch (openAiError) {
          audioText = '';
          providerErrors.push(`OpenAI: ${openAiError.message || 'failed'}`);
        }
      }

      if (!audioText && this.googleClient) {
        try {
          audioText = await this.withTimeout(this.transcribeWithGemini(audioPath), 90000, 'Gemini transcription timed out.');
          source = 'gemini-audio';
        } catch (geminiError) {
          providerErrors.push(`Gemini: ${geminiError.message || 'failed'}`);
        }
      }

      if (!audioText && this.localWhisperEnabled) {
        try {
          audioText = await this.withTimeout(this.transcribeWithLocalWhisper(audioPath), 240000, 'Local whisper transcription timed out.');
          source = 'local-whisper';
        } catch (localError) {
          providerErrors.push(`LocalWhisper: ${localError.message || 'failed'}`);
        }
      }

      if (!audioText) {
        const detail = providerErrors.length ? ` (${providerErrors.join(' | ')})` : '';
        throw new Error(`No subtitles found and audio transcription providers are unavailable or out of quota${detail}.`);
      }

      return {
        text: audioText,
        source,
        language: config.transcriptionLanguage,
        videoTitle: metadata.title,
        durationSeconds: metadata.duration || 0,
        generatedAt: new Date().toISOString()
      };
    } finally {
      if (fs.existsSync(audioPath)) {
        fs.unlinkSync(audioPath);
      }
    }
  }

  async trySubtitleExtraction(url, outputDir) {
    const baseTemplate = path.join(outputDir, 'captions');
    const language = config.transcriptionLanguage || 'en';

    const args = [
      '--skip-download',
      '--write-auto-subs',
      '--write-subs',
      '--sub-format', 'vtt',
      '--sub-langs', `${language}.*,${language},en.*,en`,
      '--sleep-requests', '1',
      '-o', `${baseTemplate}.%(ext)s`,
      '--no-playlist',
      url
    ];

    await this.runYtdlp(args);

    const files = fs.readdirSync(outputDir).filter(file => file.endsWith('.vtt'));
    if (!files.length) return null;

    const bestFile = files.find(file => file.includes(`.${language}.`)) || files[0];
    const filePath = path.join(outputDir, bestFile);
    const vttContent = fs.readFileSync(filePath, 'utf8');

    return {
      text: this.parseVttToPlainText(vttContent),
      language: this.extractLanguageFromFilename(bestFile)
    };
  }

  async downloadAudio(url, outputDir) {
    const outputTemplate = path.join(outputDir, 'audio.%(ext)s');
    const primaryArgs = [
      '-f', 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best',
      '--no-playlist',
      '--extractor-retries', '3',
      '--fragment-retries', '3',
      '--sleep-requests', '1',
      '-o', outputTemplate,
      url
    ];

    try {
      await this.runYtdlp(primaryArgs);
    } catch {
      const fallbackArgs = [
        '-f', 'bestaudio/best',
        '--no-playlist',
        '--extractor-retries', '3',
        '--fragment-retries', '3',
        '-o', outputTemplate,
        url
      ];
      await this.runYtdlp(fallbackArgs);
    }

    const files = fs.readdirSync(outputDir);
    const audioFile = files.find(file => /\.(m4a|mp3|webm|opus|aac|wav)$/i.test(file));
    if (!audioFile) {
      throw new Error('Audio extraction failed.');
    }

    return path.join(outputDir, audioFile);
  }

  async transcribeWithGemini(audioPath) {
    if (!this.googleClient) {
      throw new Error('Google AI client is not configured.');
    }

    const audioBuffer = fs.readFileSync(audioPath);
    const mimeType = this.getAudioMimeType(audioPath);
    const modelCandidates = [
      'gemini-2.0-flash',
      'gemini-1.5-flash-latest',
      'gemini-1.5-pro-latest'
    ];

    let lastError = null;
    for (const modelName of modelCandidates) {
      try {
        const model = this.googleClient.getGenerativeModel({ model: modelName });
        const result = await model.generateContent([
          {
            inlineData: {
              data: audioBuffer.toString('base64'),
              mimeType
            }
          },
          `Transcribe this audio accurately. Return only transcript text in ${config.transcriptionLanguage}.`
        ]);

        const text = result?.response?.text?.() || '';
        if (text.trim()) {
          return text.trim();
        }
      } catch (error) {
        lastError = error;
      }
    }

    throw lastError || new Error('Gemini transcription failed.');
  }

  transcribeWithLocalWhisper(audioPath) {
    const scriptPath = path.join(__dirname, '..', 'scripts', 'local_transcribe.py');

    return new Promise((resolve, reject) => {
      const args = [scriptPath, audioPath, config.transcriptionLanguage, this.whisperModel];
      const process = spawn(this.pythonPath, args);

      let stdout = '';
      let stderr = '';

      process.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(stderr.trim() || 'Local whisper process failed.'));
          return;
        }

        try {
          const parsed = JSON.parse(stdout.trim() || '{}');
          const text = (parsed.text || '').trim();
          if (!text) {
            reject(new Error('Local whisper returned empty transcript.'));
            return;
          }

          resolve(text);
        } catch (error) {
          reject(new Error(`Local whisper returned invalid output: ${error.message}`));
        }
      });

      process.on('error', (error) => {
        reject(new Error(`Local whisper startup failed: ${error.message}`));
      });
    });
  }

  runYtdlp(args) {
    return new Promise((resolve, reject) => {
      const finalArgs = ['--no-update', '--js-runtimes', 'node', ...args];
      const process = spawn(this.ytdlpPath, finalArgs);
      let stderr = '';
      const timeout = setTimeout(() => {
        process.kill('SIGTERM');
      }, 90000);

      process.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      process.on('close', (code) => {
        clearTimeout(timeout);
        if (code === 0) {
          resolve();
          return;
        }

        const message = stderr.trim().split('\n').slice(-2).join(' ').trim();
        reject(new Error(message || 'yt-dlp command failed'));
      });

      process.on('error', (error) => {
        clearTimeout(timeout);
        reject(new Error(`yt-dlp error: ${error.message}`));
      });
    });
  }

  parseVttToPlainText(vttContent) {
    const lines = vttContent.split(/\r?\n/);
    const cleaned = [];
    const seen = new Set();

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      if (trimmed === 'WEBVTT') continue;
      if (/^\d+$/.test(trimmed)) continue;
      if (trimmed.includes('-->')) continue;
      if (/^(NOTE|STYLE|REGION)/.test(trimmed)) continue;

      const plain = trimmed
        .replace(/<[^>]+>/g, '')
        .replace(/\[[^\]]+\]/g, '')
        .replace(/\s+/g, ' ')
        .trim();

      if (!plain) continue;
      if (seen.has(plain)) continue;

      seen.add(plain);
      cleaned.push(plain);
    }

    return cleaned.join(' ').trim();
  }

  extractLanguageFromFilename(filename) {
    const parts = filename.split('.');
    if (parts.length < 3) return config.transcriptionLanguage;

    const langCandidate = parts[parts.length - 3];
    return langCandidate || config.transcriptionLanguage;
  }

  getAudioMimeType(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    const map = {
      '.mp3': 'audio/mpeg',
      '.m4a': 'audio/mp4',
      '.mp4': 'audio/mp4',
      '.wav': 'audio/wav',
      '.aac': 'audio/aac',
      '.webm': 'audio/webm',
      '.opus': 'audio/ogg'
    };

    return map[ext] || 'audio/mpeg';
  }

  withTimeout(promise, timeoutMs, timeoutMessage) {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error(timeoutMessage)), timeoutMs);
      promise
        .then((result) => {
          clearTimeout(timeout);
          resolve(result);
        })
        .catch((error) => {
          clearTimeout(timeout);
          reject(error);
        });
    });
  }

}

module.exports = new TranscriptionService();
