import { useEffect, useRef, useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL;

function App() {

  const [messages, setMessages] = useState(() => {

    const savedMessages =
      localStorage.getItem("banking_chat_history");

    if (savedMessages) {
      return JSON.parse(savedMessages);
    }

    return [
      {
        role: "assistant",
        content:
          "Hello! I'm your banking assistant. How can I help you today?"
      }
    ];
  });

  const [input, setInput] = useState("");

  const [loading, setLoading] = useState(false);

  const messagesEndRef = useRef(null);


  // --------------------------------------------------
  // SAVE CHAT CONTEXT
  // --------------------------------------------------

  useEffect(() => {

    localStorage.setItem(
      "banking_chat_history",
      JSON.stringify(messages)
    );

  }, [messages]);


  // --------------------------------------------------
  // AUTO SCROLL
  // --------------------------------------------------

  useEffect(() => {

    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth"
    });

  }, [messages, loading]);


  // --------------------------------------------------
  // SEND MESSAGE
  // --------------------------------------------------

  const sendMessage = async () => {

    const userMessage = input.trim();

    if (!userMessage || loading) {
      return;
    }


    // ----------------------------------------------
    // Create user message
    // ----------------------------------------------

    const newUserMessage = {
      role: "user",
      content: userMessage
    };


    // ----------------------------------------------
    // Previous conversation
    // ----------------------------------------------

    const conversationContext = messages.map(
      (message) => ({
        role: message.role,
        content: message.content
      })
    );


    // ----------------------------------------------
    // Update UI immediately
    // ----------------------------------------------

    setMessages((previous) => [
      ...previous,
      newUserMessage
    ]);

    setInput("");
    setLoading(true);


    try {

      const response = await fetch(
        `${API_URL}/chat`,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json"
          },

          body: JSON.stringify({
            message: userMessage,

            top_k: 5,

            conversation: conversationContext
          })
        }
      );


      if (!response.ok) {

        throw new Error(
          "Failed to communicate with backend."
        );
      }


      const data = await response.json();


      // ----------------------------------------------
      // Add Gemini response
      // ----------------------------------------------

      const assistantMessage = {
        role: "assistant",
        content: data.answer,
        sources: data.sources || []
      };


      setMessages((previous) => [
        ...previous,
        assistantMessage
      ]);

    }

    catch (error) {

      console.error(error);

      setMessages((previous) => [
        ...previous,
        {
          role: "assistant",
          content:
            "Sorry, I couldn't connect to the chatbot server. Please make sure the FastAPI backend is running."
        }
      ]);

    }

    finally {

      setLoading(false);

    }
  };


  // --------------------------------------------------
  // ENTER KEY
  // --------------------------------------------------

  const handleKeyDown = (event) => {

    if (event.key === "Enter" && !event.shiftKey) {

      event.preventDefault();

      sendMessage();
    }
  };


  // --------------------------------------------------
  // CLEAR CHAT
  // --------------------------------------------------

  const clearChat = () => {

    const initialMessage = {
      role: "assistant",
      content:
        "Hello! I'm your banking assistant. How can I help you today?"
    };

    setMessages([initialMessage]);

    localStorage.removeItem(
      "banking_chat_history"
    );
  };


  return (

    <div className="app">

      {/* ------------------------------------------ */}
      {/* HEADER */}
      {/* ------------------------------------------ */}

      <header className="header">

        <div>

          <h1>
            Banking Assistant
          </h1>

          <p>
            Banking & Digital Payments
          </p>

        </div>


        <button
          className="clear-button"
          onClick={clearChat}
        >
          Clear Chat
        </button>

      </header>


      {/* ------------------------------------------ */}
      {/* CHAT AREA */}
      {/* ------------------------------------------ */}

      <main className="chat-container">

        {messages.map(
          (message, index) => (

            <div
              key={index}
              className={`message-row ${
                message.role === "user"
                  ? "user-row"
                  : "assistant-row"
              }`}
            >

              <div
                className={`message ${
                  message.role === "user"
                    ? "user-message"
                    : "assistant-message"
                }`}
              >

                <div className="message-content">

                  {message.content}

                </div>


                {/* -------------------------------- */}
                {/* SOURCES */}
                {/* -------------------------------- */}

                {message.sources &&
                  message.sources.length > 0 && (

                    <div className="sources">

                      <p className="sources-title">
                        Sources
                      </p>

                      {message.sources
                        .slice(0, 3)
                        .map(
                          (source, sourceIndex) => (

                            <div
                              key={sourceIndex}
                              className="source"
                            >

                              <strong>
                                {source.category}
                              </strong>

                              <span>
                                {
                                  source.subdomain
                                }
                              </span>

                            </div>

                          )
                        )}

                    </div>

                  )}

              </div>

            </div>

          )
        )}


        {/* -------------------------------------- */}
        {/* LOADING */}
        {/* -------------------------------------- */}

        {loading && (

          <div className="message-row assistant-row">

            <div className="message assistant-message">

              <div className="typing">

                <span></span>
                <span></span>
                <span></span>

              </div>

            </div>

          </div>

        )}


        <div ref={messagesEndRef} />

      </main>


      {/* ------------------------------------------ */}
      {/* INPUT */}
      {/* ------------------------------------------ */}

      <footer className="input-area">

        <div className="input-container">

          <textarea
            value={input}
            onChange={(event) =>
              setInput(event.target.value)
            }
            onKeyDown={handleKeyDown}
            placeholder="Ask a banking question..."
            rows="1"
          />

          <button
            onClick={sendMessage}
            disabled={
              loading ||
              !input.trim()
            }
          >
            Send
          </button>

        </div>

        <p className="input-hint">
          Press Enter to send
        </p>

      </footer>

    </div>

  );
}

export default App;