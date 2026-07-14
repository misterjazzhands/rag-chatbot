"use client";

import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useRouter } from "next/navigation";
import { auth, db } from "@/lib/firebase";
import { onAuthStateChanged, signOut, User } from "firebase/auth";
import { doc, getDoc, getDocFromCache, setDoc } from "firebase/firestore";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Array<{
    id: string;
    text: string;
    source: string;
    score: number;
  }>;
}

interface ServerStatus {
  ready: boolean;
  chunk_count: number;
  indexed_documents: string[];
}

interface ModelOption {
  id: string;
  name: string;
  provider: string;
  description: string;
  host_available: boolean;
  byok_supported: boolean;
}

interface Toast {
  id: string;
  message: string;
  type: "success" | "error" | "info";
}

interface ConfirmModalState {
  isOpen: boolean;
  title: string;
  message: string;
  onConfirm: () => void;
}

export default function RAGChatbot() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<ServerStatus>({
    ready: false,
    chunk_count: 0,
    indexed_documents: [],
  });
  const [error, setError] = useState<string | null>(null);

  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [, setInitialLoadDone] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [models, setModels] = useState<ModelOption[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");

  // Toast notifications state
  const [toasts, setToasts] = useState<Toast[]>([]);

  // Custom confirmation modal state
  const [confirmModal, setConfirmModal] = useState<ConfirmModalState>({
    isOpen: false,
    title: "",
    message: "",
    onConfirm: () => { },
  });

  const chatEndRef = useRef<HTMLDivElement>(null);

  const rawBackendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000";
  const BACKEND_URL = rawBackendUrl.replace(/\/+$/, "");

  const showToast = (message: string, type: "success" | "error" | "info" = "info") => {
    const id = Math.random().toString(36).substring(2, 9);
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4500);
  };

  const fetchStatus = React.useCallback(async () => {
    if (!user) return;
    try {
      const res = await fetch(`${BACKEND_URL}/api/status`, {
        headers: { "X-User-ID": user.uid }
      });
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (err) {
      console.error("Failed to connect to RAG backend:", err);
    }
  }, [BACKEND_URL, user]);

  const fetchModels = React.useCallback(async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/models`);
      if (res.ok) {
        const data = await res.json();
        setModels(data.models);
        if (data.default) {
          setSelectedModel(data.default);
        } else if (data.models.length > 0) {
          setSelectedModel(data.models[0].id);
        }
      }
    } catch (err) {
      console.error("Failed to fetch models:", err);
    }
  }, [BACKEND_URL]);

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      if (!currentUser) {
        setMessages([]);
        setUser(null);
        setInitialLoadDone(false);
        router.push("/login");
      } else {
        setUser(currentUser);
      }
    });
    return () => unsubscribe();
  }, [router]);

  useEffect(() => {
    const init = async () => {
      await fetchModels();
      if (user) {
        await fetchStatus();
      }
    };
    init();

    const interval = setInterval(() => {
      if (user) fetchStatus();
    }, 10000);
    return () => clearInterval(interval);
  }, [fetchModels, fetchStatus, user]);

  useEffect(() => {
    const loadInitialMessages = async () => {
      if (!user) return;
      const docRef = doc(db, "users", user.uid, "chat", "history");
      try {
        // First try server fetch (default behavior)
        const docSnap = await getDoc(docRef);
        if (docSnap.exists() && docSnap.data().messages) {
          setMessages(docSnap.data().messages);
        } else {
          setMessages([]);
        }
      } catch {
        // Server fetch failed (maybe offline). Try cache.
        try {
          const cacheSnap = await getDocFromCache(docRef);
          if (cacheSnap.exists() && cacheSnap.data().messages) {
            setMessages(cacheSnap.data().messages);
          } else {
            setMessages([]);
          }
        } catch {
          // No cached document; start with empty chat.
          setMessages([]);
        }
      } finally {
        setInitialLoadDone(true);
      }
    };
    loadInitialMessages();
  }, [user]);

  const syncChatHistory = (newMessages: Message[]) => {
    if (!user) return;
    const docRef = doc(db, "users", user.uid, "chat", "history");
    setDoc(docRef, { messages: newMessages }, { merge: true }).catch(console.error);
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0 || !user) return;

    const file = files[0];
    if (file.type !== "application/pdf") {
      showToast("Please upload PDF files only.", "error");
      return;
    }

    setUploading(true);
    setError(null);

    try {
      // Direct upload to backend - no Firebase Storage needed
      const formData = new FormData();
      formData.append("file", file);

      const requestUrl = `${BACKEND_URL}/api/upload`;
      const res = await fetch(requestUrl, {
        method: "POST",
        headers: {
          "x-user-id": user.uid,
          // DO NOT set Content-Type - browser sets it automatically with boundary
        },
        body: formData,
      });

      if (!res.ok) {
        const errText = await res.text();
        let errMsg = "Upload failed";
        try {
          const errData = JSON.parse(errText);
          errMsg = errData.detail || errMsg;
        } catch (e) {
          errMsg = `Server error (${res.status}) on ${requestUrl}: ` + errText.substring(0, 100);
        }
        throw new Error(errMsg);
      }

      await fetchStatus();
      showToast(`Successfully uploaded and indexed: ${file.name}`, "success");
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "An error occurred during indexing.";
      setError(errorMessage);
      showToast(errorMessage, "error");
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  const handleResetClick = () => {
    setConfirmModal({
      isOpen: true,
      title: "Clear Vector Database",
      message: "Are you sure you want to delete all indexed document chunks? This will wipe the vector database collection and uploads folder, and cannot be undone.",
      onConfirm: async () => {
        setConfirmModal((prev) => ({ ...prev, isOpen: false }));
        try {
          const res = await fetch(`${BACKEND_URL}/api/reset`, {
            method: "POST",
            headers: { "X-User-ID": user?.uid || "" }
          });
          if (res.ok) {
            setMessages([]);
            syncChatHistory([]);
            setStatus({ ready: false, chunk_count: 0, indexed_documents: [] });
            showToast("Database and chat history cleared successfully!", "success");
          } else {
            throw new Error("Failed to clear database");
          }
        } catch {
          setError("Failed to reset database.");
          showToast("Failed to reset vector database.", "error");
        }
      }
    });
  };

  const handleClearChat = () => {
    setConfirmModal({
      isOpen: true,
      title: "Clear Chat History",
      message: "Are you sure you want to clear your current conversation history? The vector database will remain intact.",
      onConfirm: () => {
        setMessages([]);
        syncChatHistory([]);
        setConfirmModal((prev) => ({ ...prev, isOpen: false }));
        showToast("Conversation history cleared.", "info");
      }
    });
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userMessage: Message = { role: "user", content: query };
    setMessages((prev) => [...prev, userMessage]);
    setQuery("");
    setLoading(true);
    setError(null);

    const historyPayload = messages.map(({ role, content }) => ({ role, content }));

    try {
      const res = await fetch(`${BACKEND_URL}/api/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-User-ID": user?.uid || ""
        },
        body: JSON.stringify({
          query: userMessage.content,
          chat_history: historyPayload,
          model_id: selectedModel || undefined,
        }),
      });

      if (!res.ok) {
        let errMsg = "Server error";
        try {
          const errData = await res.json();
          errMsg = errData.detail || errMsg;
        } catch {
          // Ignore json parse error
        }
        throw new Error(errMsg);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No reader available");

      const decoder = new TextDecoder("utf-8");
      let buffer = "";

      setMessages((prev) => [...prev, { role: "assistant", content: "", sources: [] }]);

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const cleanedLine = line.trim();
          if (!cleanedLine.startsWith("data: ")) continue;

          try {
            const payload = JSON.parse(cleanedLine.substring(6));
            if (payload.type === "sources") {
              setMessages((prev) => {
                const next = [...prev];
                const last = next[next.length - 1];
                if (last && last.role === "assistant") {
                  next[next.length - 1] = { ...last, sources: payload.sources };
                }
                return next;
              });
            } else if (payload.type === "token") {
              setMessages((prev) => {
                const next = [...prev];
                const last = next[next.length - 1];
                if (last && last.role === "assistant") {
                  next[next.length - 1] = { ...last, content: last.content + payload.token };
                }
                return next;
              });
            } else if (payload.type === "error") {
              throw new Error(payload.error);
            }
          } catch (e) {
            console.error("Error parsing stream chunk:", e);
          }
        }
      }

      // Stream finished successfully, sync the final state to Firestore
      setMessages((prev) => {
        syncChatHistory(prev);
        return prev;
      });
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Query failed. Please try again.";
      setError(errorMessage);
      showToast(errorMessage, "error");
      setMessages((prev) => {
        const next = [...prev];
        const last = next[next.length - 1];
        if (last && last.role === "assistant" && last.content === "") {
          next.pop();
        }
        syncChatHistory(next);
        return next;
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[var(--color-canvas)] text-[var(--color-ink)]">

      {/* 🚀 Apple Global Nav */}
      <header className="h-[44px] bg-[var(--color-surface-black)] text-[var(--color-on-dark)] px-6 flex justify-between items-center shrink-0 w-full z-10">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full ${status.ready ? 'bg-[#34c759]' : 'bg-[#ff9f0a]'}`} />
            <span className="text-[12px] font-[400] tracking-[-0.12px]">
              {status.ready ? `${status.chunk_count} Chunks Loaded` : "No Knowledge Base"}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-4 relative">
          <button
            onClick={() => setIsProfileOpen(!isProfileOpen)}
            className="w-[32px] h-[32px] rounded-full bg-[var(--color-surface-pearl)] border border-[var(--color-hairline)] flex items-center justify-center text-[var(--color-ink)] hover:bg-[var(--color-canvas-parchment)] transition-colors cursor-pointer overflow-hidden"
          >
            {user?.photoURL ? (
              <>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={user.photoURL} alt="Profile" className="w-full h-full object-cover" />
              </>
            ) : (
              <span className="text-[14px] font-semibold text-[var(--color-ink)]">
                {user?.displayName ? user.displayName.charAt(0).toUpperCase() : user?.email ? user.email.charAt(0).toUpperCase() : "U"}
              </span>
            )}
          </button>

          {isProfileOpen && (
            <div className="absolute top-[44px] right-0 w-[240px] bg-[var(--color-canvas)] border border-[var(--color-hairline)] rounded-[11px] shadow-[rgba(0,0,0,0.1)_0px_10px_30px] p-2 z-50 flex flex-col gap-1 animate-in fade-in zoom-in-95 duration-150">
              <div className="px-3 py-2 border-b border-[var(--color-divider-soft)] mb-1">
                <div className="caption-text text-[var(--color-ink-muted-80)]">Signed in as</div>
                <div className="body-text font-[600] text-[var(--color-ink)] truncate">
                  {user?.displayName ? user.displayName : (user?.email ? user.email : "Anonymous User")}
                </div>
              </div>
              <button
                onClick={() => { handleClearChat(); setIsProfileOpen(false); }}
                disabled={messages.length === 0}
                className="text-left px-3 py-2 text-[14px] text-[var(--color-ink)] hover:bg-[var(--color-canvas-parchment)] rounded-[8px] disabled:opacity-40 disabled:cursor-not-allowed transition-colors cursor-pointer"
              >
                Clear Chat
              </button>
              <button
                onClick={() => { handleResetClick(); setIsProfileOpen(false); }}
                className="text-left px-3 py-2 text-[14px] text-[var(--color-ink)] hover:bg-[var(--color-canvas-parchment)] rounded-[8px] transition-colors cursor-pointer"
              >
                Clear Vector DB
              </button>
              <div className="h-[1px] bg-[var(--color-divider-soft)] my-1"></div>
              <button
                onClick={() => signOut(auth)}
                className="text-left px-3 py-2 text-[14px] text-[#ff3b30] hover:bg-[#fff0f0] rounded-[8px] transition-colors cursor-pointer"
              >
                Logout
              </button>
            </div>
          )}
        </div>
      </header>

      <main className="flex flex-1 overflow-hidden">

        {/* Left Panel - Knowledge Ingestion (Parchment) */}
        <section className="w-80 bg-[var(--color-canvas-parchment)] border-r border-[var(--color-hairline)] p-8 flex flex-col gap-6 overflow-y-auto hidden md:flex shrink-0">
          <div>
            <h2 className="display-lg text-[34px] tracking-tight">Library</h2>
            <p className="body-text mt-2 text-[var(--color-ink-muted-80)]">
              Upload PDF documents to build the local knowledge base.
            </p>
          </div>

          <div className="bg-[var(--color-canvas)] border border-[var(--color-hairline)] rounded-[18px] p-6 flex flex-col items-center justify-center text-center cursor-pointer transition-all relative min-h-[140px]">
            <input
              type="file"
              accept=".pdf"
              onChange={handleFileUpload}
              disabled={uploading}
              className="absolute inset-0 opacity-0 cursor-pointer"
            />
            {uploading ? (
              <div className="flex flex-col items-center gap-2 body-text text-[var(--color-primary)]">
                <span className="animate-spin text-xl">↻</span>
                <span>Indexing...</span>
              </div>
            ) : (
              <div className="body-text text-[var(--color-ink)] flex flex-col items-center gap-1">
                <span className="text-[var(--color-primary)]">Select PDF Document</span>
                <span className="caption-text text-[var(--color-ink-muted-80)]">or drag and drop</span>
              </div>
            )}
          </div>

          {status.indexed_documents.length > 0 && (
            <div className="flex-grow overflow-y-auto mt-4">
              <h3 className="body-strong mb-3 text-[var(--color-ink)]">
                Indexed Files
              </h3>
              <div className="space-y-3">
                {status.indexed_documents.map((doc, idx) => (
                  <div key={idx} className="body-text text-[var(--color-ink-muted-80)] flex items-center gap-3">
                    <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
                    </svg>
                    <span className="truncate">{doc}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* Right Panel - Chat Feed (Pure White) */}
        <section className="flex-grow flex flex-col bg-[var(--color-canvas)] overflow-hidden relative">

          {error && (
            <div className="mx-auto mt-6 w-full max-w-2xl px-4">
              <div className="p-4 bg-[var(--color-surface-pearl)] border border-[var(--color-hairline)] rounded-[11px] caption-text flex justify-between items-center">
                <span className="text-[#ff3b30]">There was an error: {error}</span>
                <button onClick={() => setError(null)} className="text-[var(--color-ink-muted-80)] hover:text-[var(--color-ink)]">✕</button>
              </div>
            </div>
          )}

          <div className="flex-1 overflow-y-auto px-4 py-8 space-y-8 scroll-smooth">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center max-w-lg mx-auto gap-4 px-6">
                <h3 className="hero-display text-[40px] tracking-tight">How can I help?</h3>
                <p className="body-text text-[var(--color-ink-muted-80)]">
                  {status.ready
                    ? "Ask questions about your uploaded documents, and I'll generate precise grounded answers."
                    : "Please upload a document to begin."}
                </p>
              </div>
            ) : (
              <div className="max-w-3xl mx-auto space-y-8">
                {messages.map((msg, index) => (
                  <div key={index} className={`flex flex-col ${msg.role === "user" ? "items-end" : "items-start"}`}>

                    {/* Message Bubble */}
                    {msg.role === "user" ? (
                      <div className="bg-[var(--color-surface-tile-1)] text-[var(--color-on-dark)] px-5 py-3 rounded-[18px] body-text max-w-[80%] inline-block">
                        {msg.content}
                      </div>
                    ) : (
                      <div className="w-full text-[var(--color-ink)] py-2">
                        <div className="prose-markdown">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {msg.content || "Thinking..."}
                          </ReactMarkdown>
                        </div>

                        {/* Sources */}
                        {msg.sources && msg.sources.length > 0 && (
                          <div className="mt-6 pt-6 border-t border-[var(--color-divider-soft)]">
                            <span className="caption-text font-[600] block mb-3">References</span>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                              {msg.sources.map((src, sIdx) => (
                                <div key={sIdx} className="bg-[var(--color-surface-pearl)] border border-[var(--color-hairline)] rounded-[11px] p-4 text-[12px] leading-tight">
                                  <div className="font-[600] mb-2 truncate text-[var(--color-ink)]">[{sIdx + 1}] {src.id.split("/").pop()}</div>
                                  <div className="text-[var(--color-ink-muted-80)] line-clamp-3">&quot;{src.text}&quot;</div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                  </div>
                ))}

                {loading && messages.length > 0 && messages[messages.length - 1].content === "" && (
                  <div className="flex gap-2 items-center body-text text-[var(--color-ink-muted-80)] animate-pulse">
                    Generating...
                  </div>
                )}
                <div ref={chatEndRef} />
              </div>
            )}
          </div>

          {/* Form Input */}
          <div className="p-6 bg-[var(--color-canvas)]">
            <div className="max-w-3xl mx-auto flex flex-col gap-3">
              {models.length > 0 && (
                <div className="flex items-center gap-2 px-1">
                  <label htmlFor="modelSelect" className="caption-text font-semibold text-[var(--color-ink-muted-80)]">
                    Model:
                  </label>
                  <select
                    id="modelSelect"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    disabled={!status.ready || loading}
                    className="caption-text bg-[var(--color-surface-pearl)] text-[var(--color-ink)] border border-[var(--color-hairline)] rounded-[8px] px-2 py-1 outline-none focus:ring-1 focus:ring-[var(--color-primary)] transition-shadow cursor-pointer disabled:opacity-50"
                  >
                    {models.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name} {m.host_available ? "" : "(Needs API Key)"}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              <form onSubmit={handleSendMessage} className="flex gap-4">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  disabled={!status.ready || loading}
                  placeholder={status.ready ? "Ask something about your uploaded files" : "Index a document first"}
                  className="apple-search-input flex-grow disabled:opacity-50"
                />
                <button
                  type="submit"
                  disabled={!status.ready || loading || !query.trim()}
                  className="apple-button-primary shrink-0 disabled:opacity-50 disabled:bg-[var(--color-divider-soft)] disabled:text-[var(--color-ink-muted-48)]"
                >
                  Ask
                </button>
              </form>
            </div>
          </div>

        </section>
      </main>

      {/* Toasts */}
      <div className="fixed top-12 right-6 z-50 flex flex-col gap-3 pointer-events-none max-w-sm w-full">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className="p-4 bg-[var(--color-surface-pearl)] rounded-[11px] border border-[var(--color-hairline)] shadow-[rgba(0,0,0,0.1)_0px_10px_30px] flex items-center justify-between pointer-events-auto"
          >
            <span className="body-text text-[var(--color-ink)]">{toast.message}</span>
            <button
              onClick={() => setToasts((prev) => prev.filter((t) => t.id !== toast.id))}
              className="ml-4 text-[var(--color-ink-muted-80)]"
            >
              ✕
            </button>
          </div>
        ))}
      </div>

      {/* Confirmation Modal */}
      {confirmModal.isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[rgba(0,0,0,0.4)] backdrop-blur-sm p-4">
          <div className="bg-[var(--color-canvas)] border border-[var(--color-hairline)] rounded-[18px] max-w-md w-full p-8 shadow-[rgba(0,0,0,0.22)_3px_5px_30px_0px] animate-in fade-in zoom-in duration-200">
            <h3 className="body-strong text-[21px] mb-3">
              {confirmModal.title}
            </h3>
            <p className="body-text text-[var(--color-ink-muted-80)] mb-8">
              {confirmModal.message}
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setConfirmModal((prev) => ({ ...prev, isOpen: false }))}
                className="apple-button-dark-utility bg-[var(--color-surface-pearl)] text-[var(--color-ink)] border border-[var(--color-hairline)]"
              >
                Cancel
              </button>
              <button
                onClick={confirmModal.onConfirm}
                className="apple-button-primary bg-[#ff3b30] hover:bg-[#ff453a]"
              >
                Confirm
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}
