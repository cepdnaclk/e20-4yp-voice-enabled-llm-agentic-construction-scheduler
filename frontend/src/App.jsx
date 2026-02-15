import { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import ChatInterface from './components/ChatInterface';
import TaskBoard from './components/TaskBoard';
import StageTracker from './components/StageTracker';
import { startChat, sendMessage, resumeChat } from './api';

function App() {
  const [threadId, setThreadId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [currentStage, setCurrentStage] = useState('intent');
  const [phases, setPhases] = useState([]);
  const [generatedTasks, setGeneratedTasks] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [pendingInterrupt, setPendingInterrupt] = useState(null);
  const [isInitializing, setIsInitializing] = useState(true);
  const streamingMessageRef = useRef('');

  // Initialize chat on mount
  useEffect(() => {
    initializeChat();
  }, []);

  const initializeChat = async () => {
    try {
      setIsInitializing(true);
      const data = await startChat();
      setThreadId(data.thread_id);

      if (data.message) {
        setMessages([{ role: 'ai', content: data.message }]);
      }
      if (data.interrupt) {
        setPendingInterrupt(data.interrupt);
      }
      if (data.stage) {
        setCurrentStage(data.stage);
      }
    } catch (err) {
      console.error('Failed to start chat:', err);
      setMessages([{ role: 'ai', content: '⚠️ Failed to connect to the server. Make sure the backend is running on port 8000.' }]);
    } finally {
      setIsInitializing(false);
    }
  };

  const getSSECallbacks = useCallback(() => ({
    onChunk: (chunk) => {
      streamingMessageRef.current += chunk;
      setMessages(prev => {
        const updated = [...prev];
        const lastMsg = updated[updated.length - 1];
        if (lastMsg && lastMsg.role === 'ai' && lastMsg.streaming) {
          updated[updated.length - 1] = {
            ...lastMsg,
            content: streamingMessageRef.current
          };
        }
        return updated;
      });
    },
    onInterrupt: (value) => {
      setPendingInterrupt(value);
    },
    onStageChange: (stage) => {
      setCurrentStage(stage);
    },
    onTasks: (tasks) => {
      setGeneratedTasks(tasks);
    },
    onPhases: (phasesData) => {
      setPhases(phasesData);
    },
    onDone: () => {
      setMessages(prev => {
        const updated = [...prev];
        const lastMsg = updated[updated.length - 1];
        if (lastMsg && lastMsg.streaming) {
          updated[updated.length - 1] = { ...lastMsg, streaming: false };
        }
        return updated;
      });
      setIsLoading(false);
    },
    onError: (err) => {
      console.error('Stream error:', err);
      setIsLoading(false);
    },
  }), []);

  const handleSend = async (text) => {
    if (!text.trim() || !threadId || isLoading) return;

    // Add user message
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setIsLoading(true);
    streamingMessageRef.current = '';

    // Add empty streaming AI message
    setMessages(prev => [...prev, { role: 'ai', content: '', streaming: true }]);

    if (pendingInterrupt) {
      // Resume from interrupt
      setPendingInterrupt(null);
      await resumeChat(threadId, text, getSSECallbacks());
    } else {
      // Normal message
      await sendMessage(threadId, text, getSSECallbacks());
    }
  };

  const showTaskBoard = currentStage === 'details' || currentStage === 'scheduling';

  return (
    <div className="app">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <div className="logo">
            <span className="logo-icon">🏗️</span>
            <h1>Construction Planner</h1>
          </div>
          <span className="header-subtitle">AI-Powered Schedule Generator</span>
        </div>
        <StageTracker currentStage={currentStage} />
      </header>

      {/* Main Content */}
      <main className={`app-main ${showTaskBoard ? 'with-tasks' : ''}`}>
        <div className="chat-panel">
          <ChatInterface
            messages={messages}
            onSend={handleSend}
            isLoading={isLoading}
            isInitializing={isInitializing}
            pendingInterrupt={pendingInterrupt}
            currentStage={currentStage}
          />
        </div>

        {showTaskBoard && (
          <div className="task-panel">
            <TaskBoard
              phases={phases}
              generatedTasks={generatedTasks}
            />
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
