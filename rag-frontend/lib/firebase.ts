import { initializeApp, getApps, getApp } from "firebase/app";
import { getAuth, GoogleAuthProvider } from "firebase/auth";
import { initializeFirestore } from "firebase/firestore";
import { getStorage } from "firebase/storage";

const firebaseConfig = {
  apiKey: "AIzaSyDIFNNzameE9Yu7ZYvvxHYd_AuaQmmfYmE",
  authDomain: "rag-chatbot-1-57e5a.firebaseapp.com",
  projectId: "rag-chatbot-1-57e5a",
  storageBucket: "rag-chatbot-1-57e5a.firebasestorage.app",
  messagingSenderId: "1091802973983",
  appId: "1:1091802973983:web:f4df0c15ad4948fff1122a",
  measurementId: "G-B5Z8DWT007"
};

// Initialize Firebase only if it hasn't been initialized yet
const app = !getApps().length ? initializeApp(firebaseConfig) : getApp();
const auth = getAuth(app);
const googleProvider = new GoogleAuthProvider();

// Use initializeFirestore with long polling to prevent "client is offline" errors with Next.js Turbopack
const db = initializeFirestore(app, { experimentalForceLongPolling: true });
const storage = getStorage(app);

export { app, auth, googleProvider, db, storage };
