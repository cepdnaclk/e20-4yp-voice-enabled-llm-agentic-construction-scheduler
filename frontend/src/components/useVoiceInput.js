import { useState, useRef, useCallback, useEffect } from 'react';

/**
 * useVoiceInput — React hook wrapping the Web Speech API.
 *
 * Real-time transcription: interim results appear in the textarea as you speak.
 * Auto-submit: fires only when the user explicitly clicks the mic button to stop
 *             (NOT on browser-initiated pauses), so half-sentences are never sent.
 *
 * Works across ALL agent stages — intent, phase, details, scheduling —
 * because transcribed text goes through the same onSend() path as typed text.
 *
 * @param {React.RefObject} inputRef     - ref to the <textarea>
 * @param {Function}        onAutoSubmit - optional: called with final text on explicit stop
 */
function useVoiceInput(inputRef, onAutoSubmit = null) {
    const [isListening, setIsListening] = useState(false);
    const recognitionRef  = useRef(null);
    const finalRef        = useRef('');          // accumulates confirmed words
    const userStoppedRef  = useRef(false);       // true when USER clicked stop (not browser)
    const autoSubmitRef   = useRef(onAutoSubmit);

    // Keep the callback ref fresh without destroying the recognition instance
    useEffect(() => {
        autoSubmitRef.current = onAutoSubmit;
    }, [onAutoSubmit]);

    const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
    const isSupported = Boolean(SpeechRecognition);

    // ── Write text directly into the uncontrolled textarea ────────────────────
    const writeToTextarea = (text) => {
        const el = inputRef.current;
        if (!el) return;

        // Direct DOM write — works for uncontrolled <textarea>
        el.value = text;

        // Dispatch 'input' so the auto-resize onInput handler in ChatInterface fires
        el.dispatchEvent(new Event('input', { bubbles: true }));

        // Manual resize fallback
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 150) + 'px';
    };

    // ── Build the recognition instance once ───────────────────────────────────
    useEffect(() => {
        if (!isSupported) return;

        const recognition = new SpeechRecognition();
        recognition.continuous     = true;   // keep going until explicitly stopped
        recognition.interimResults = true;   // fire onresult with partial words in real time
        recognition.lang           = 'en-US';

        recognition.onstart = () => {
            setIsListening(true);
            finalRef.current       = '';
            userStoppedRef.current = false;
        };

        // ── This fires for EVERY word (interim) and confirmed sentence (final) ──
        recognition.onresult = (event) => {
            let interim = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalRef.current += transcript + ' ';
                } else {
                    interim = transcript;          // live word-by-word preview
                }
            }

            // Show confirmed words + current in-progress word
            writeToTextarea(finalRef.current + interim);
        };

        recognition.onerror = (event) => {
            if (event.error !== 'no-speech' && event.error !== 'aborted') {
                console.warn('SpeechRecognition error:', event.error);
            }
            setIsListening(false);
        };

        recognition.onend = () => {
            setIsListening(false);

            const text = finalRef.current.trim();

            // Auto-submit ONLY if the USER clicked stop (not a browser-initiated end)
            if (userStoppedRef.current && autoSubmitRef.current && text) {
                userStoppedRef.current = false;
                // Small delay so the textarea shows final text before it's cleared
                setTimeout(() => {
                    autoSubmitRef.current(text);
                    writeToTextarea('');
                    finalRef.current = '';
                }, 100);
            }
        };

        recognitionRef.current = recognition;

        return () => { recognition.abort(); };
    }, [isSupported]); // eslint-disable-line react-hooks/exhaustive-deps

    // ── Toggle: start or stop recording ───────────────────────────────────────
    const toggleListening = useCallback(() => {
        const rec = recognitionRef.current;
        if (!rec) return;

        if (isListening) {
            userStoppedRef.current = true;   // mark as user-initiated stop
            rec.stop();
        } else {
            // Clear textarea for a fresh recording session
            writeToTextarea('');
            finalRef.current = '';
            userStoppedRef.current = false;
            try { rec.start(); } catch { /* already started */ }
        }
    }, [isListening]); // eslint-disable-line react-hooks/exhaustive-deps

    return { isListening, isSupported, toggleListening };
}

export default useVoiceInput;
