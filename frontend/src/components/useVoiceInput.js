import { useState, useRef, useCallback, useEffect } from 'react';

/**
 * useVoiceInput — React hook that wraps the Web Speech API.
 *
 * @param {React.RefObject} inputRef - ref to the textarea to inject transcripts into
 * @returns {{ isListening, isSupported, toggleListening }}
 */
function useVoiceInput(inputRef) {
    const [isListening, setIsListening] = useState(false);
    const recognitionRef = useRef(null);

    // Check browser support (Chrome/Edge: webkitSpeechRecognition, newer spec: SpeechRecognition)
    const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
    const isSupported = Boolean(SpeechRecognition);

    // Build the recognition instance once
    useEffect(() => {
        if (!isSupported) return;

        const recognition = new SpeechRecognition();
        recognition.continuous = true;       // keep listening until explicitly stopped
        recognition.interimResults = true;   // show partial results as user speaks
        recognition.lang = 'en-US';

        let finalTranscript = '';

        recognition.onstart = () => {
            setIsListening(true);
            finalTranscript = ''; // reset accumulator on each session
        };

        recognition.onresult = (event) => {
            let interim = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalTranscript += transcript + ' ';
                } else {
                    interim = transcript;
                }
            }

            if (inputRef.current) {
                // Combine finalised words + live interim text
                inputRef.current.value = finalTranscript + interim;

                // Trigger React's synthetic onChange so the component knows the value changed
                const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
                    window.HTMLTextAreaElement.prototype,
                    'value'
                ).set;
                nativeInputValueSetter.call(inputRef.current, inputRef.current.value);
                inputRef.current.dispatchEvent(new Event('input', { bubbles: true }));

                // Auto-resize the textarea
                inputRef.current.style.height = 'auto';
                inputRef.current.style.height =
                    Math.min(inputRef.current.scrollHeight, 150) + 'px';
            }
        };

        recognition.onerror = (event) => {
            console.warn('SpeechRecognition error:', event.error);
            setIsListening(false);
        };

        recognition.onend = () => {
            setIsListening(false);
        };

        recognitionRef.current = recognition;

        return () => {
            recognition.abort();
        };
    }, [isSupported]); // eslint-disable-line react-hooks/exhaustive-deps

    const toggleListening = useCallback(() => {
        if (!recognitionRef.current) return;

        if (isListening) {
            recognitionRef.current.stop();
        } else {
            // Clear old text so each recording session starts fresh (optional UX choice)
            // Remove the two lines below if you want to APPEND to existing text instead
            if (inputRef.current) {
                inputRef.current.value = '';
                inputRef.current.style.height = 'auto';
            }
            try {
                recognitionRef.current.start();
            } catch (err) {
                // Ignore "already started" errors
            }
        }
    }, [isListening, inputRef]);

    return { isListening, isSupported, toggleListening };
}

export default useVoiceInput;
