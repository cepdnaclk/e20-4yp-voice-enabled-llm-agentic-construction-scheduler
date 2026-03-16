import { useState, useRef, useCallback, useEffect } from 'react';

/**
 * useWhisperInput — Records audio via MediaRecorder and transcribes it
 * with Faster Whisper running on the backend (/api/transcribe).
 *
 * Flow:
 *   1. User clicks mic → MediaRecorder starts capturing audio
 *   2. Words appear in textarea in real-time via interim audio chunks
 *      (sent every CHUNK_INTERVAL ms while recording)
 *   3. User clicks mic again → recording stops → full audio sent for final transcription
 *   4. If onAutoSubmit is provided, the final text is automatically sent
 *
 * Works for all agent stages (intent / phase / details / scheduling) because
 * transcribed text flows through the same onSend() path as typed text.
 *
 * @param {React.RefObject} inputRef     - ref to the <textarea>
 * @param {Function}        onAutoSubmit - optional: called with final text on stop
 */

const BACKEND_URL = 'http://localhost:8000';
const CHUNK_INTERVAL_MS = 1500;  // send interim chunk every 1.5s for live preview

function useWhisperInput(inputRef, onAutoSubmit = null) {
    const [isListening, setIsListening]     = useState(false);
    const [isTranscribing, setIsTranscribing] = useState(false);
    const [isSupported, setIsSupported]     = useState(false);

    const mediaRecorderRef = useRef(null);
    const chunksRef        = useRef([]);
    const streamRef        = useRef(null);
    const intervalRef      = useRef(null);
    const accumulatedRef   = useRef('');   // confirmed text so far
    const baseTextRef      = useRef('');   // text that existed before current recording session
    const autoSubmitRef    = useRef(onAutoSubmit);

    useEffect(() => { autoSubmitRef.current = onAutoSubmit; }, [onAutoSubmit]);

    // Check if MediaRecorder + getUserMedia are available
    useEffect(() => {
        setIsSupported(
            typeof window !== 'undefined' &&
            !!navigator.mediaDevices?.getUserMedia &&
            typeof MediaRecorder !== 'undefined'
        );
    }, []);

    // ── Write text into the uncontrolled textarea ────────────────────────────
    const writeToTextarea = useCallback((text) => {
        const el = inputRef.current;
        if (!el) return;
        el.value = text;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 150) + 'px';
    }, [inputRef]);

    const combineText = useCallback((baseText, appendText) => {
        const base = (baseText || '').trim();
        const append = (appendText || '').trim();
        if (!base) return append;
        if (!append) return base;
        return `${base} ${append}`;
    }, []);

    // ── Send audio blob to backend Whisper endpoint ──────────────────────────
    const transcribeBlob = useCallback(async (blob, isInterim = false) => {
        try {
            const form = new FormData();
            form.append('audio', blob, 'recording.webm');
            
            const endpoint = isInterim ? '/api/transcribe/interim' : '/api/transcribe';
            const res = await fetch(`${BACKEND_URL}${endpoint}`, {
                method: 'POST',
                body: form,
            });

            if (!res.ok) throw new Error(`Transcription failed: ${res.status}`);
            const data = await res.json();
            return (data.text || '').trim();
        } catch (err) {
            console.error('Whisper transcription error:', err);
            return '';
        }
    }, []);

    // ── Send an interim chunk for live preview ───────────────────────────────
    const sendInterimChunk = useCallback(async () => {
        if (chunksRef.current.length === 0) return;

        // Snapshot current chunks without stopping the recorder
        const snapshot = new Blob([...chunksRef.current], { type: 'audio/webm' });
        const text = await transcribeBlob(snapshot, true); // true = use fast endpoint

        if (text) {
            accumulatedRef.current = text;
            const previewText = combineText(baseTextRef.current, text);
            writeToTextarea(previewText + ' ...');  // trailing dots = still recording
        }
    }, [combineText, transcribeBlob, writeToTextarea]);

    // ── Start recording ──────────────────────────────────────────────────────
    const startListening = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            streamRef.current = stream;

            const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                ? 'audio/webm;codecs=opus'
                : 'audio/webm';

            const recorder = new MediaRecorder(stream, { mimeType });
            mediaRecorderRef.current = recorder;
            chunksRef.current = [];
            accumulatedRef.current = '';
            baseTextRef.current = (inputRef.current?.value || '').trim();

            recorder.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data);
            };

            recorder.start(500);  // collect a chunk every 500ms
            setIsListening(true);

            // Send interim chunks every CHUNK_INTERVAL_MS for live preview
            intervalRef.current = setInterval(sendInterimChunk, CHUNK_INTERVAL_MS);

        } catch (err) {
            console.error('Microphone access error:', err);
            alert('Could not access microphone. Please allow microphone permissions.');
        }
    }, [inputRef, sendInterimChunk]);

    // ── Stop recording & finalize ────────────────────────────────────────────
    const stopListening = useCallback(() => {
        clearInterval(intervalRef.current);
        intervalRef.current = null;

        setIsListening(false);

        const recorder = mediaRecorderRef.current;
        if (!recorder) return;

        recorder.onstop = async () => {
            setIsTranscribing(true);

            if (accumulatedRef.current) {
                writeToTextarea(combineText(baseTextRef.current, accumulatedRef.current));
            }

            const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
            const finalText = await transcribeBlob(blob);

            setIsTranscribing(false);

            if (finalText) {
                const mergedText = combineText(baseTextRef.current, finalText);
                writeToTextarea(mergedText);
                // Auto-submit after a short delay so user sees the text first
                if (autoSubmitRef.current) {
                    setTimeout(() => {
                        autoSubmitRef.current(mergedText);
                        writeToTextarea('');
                    }, 500);
                }
            }

            // Release microphone
            streamRef.current?.getTracks().forEach(t => t.stop());
            streamRef.current = null;
            mediaRecorderRef.current = null;
            chunksRef.current = [];
        };

        recorder.stop();
    }, [combineText, transcribeBlob, writeToTextarea]);

    // ── Toggle on mic button click ───────────────────────────────────────────
    const toggleListening = useCallback(() => {
        if (isListening) {
            stopListening();
        } else {
            startListening();
        }
    }, [isListening, startListening, stopListening]);

    // ── Cleanup on unmount ───────────────────────────────────────────────────
    useEffect(() => {
        return () => {
            clearInterval(intervalRef.current);
            streamRef.current?.getTracks().forEach(t => t.stop());
        };
    }, []);

    return { isListening, isTranscribing, isSupported, toggleListening };
}

export default useWhisperInput;
