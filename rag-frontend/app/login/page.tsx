"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { signInWithEmailAndPassword, createUserWithEmailAndPassword, signInWithPopup } from "firebase/auth";
import { auth, googleProvider } from "@/lib/firebase";

export default function LoginPage() {
  const router = useRouter();
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  const handleEmailAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    try {
      if (isLogin) {
        await signInWithEmailAndPassword(auth, email, password);
      } else {
        await createUserWithEmailAndPassword(auth, email, password);
      }
      router.push("/");
    } catch (err: unknown) {
      setError((err as Error).message);
    }
  };

  const handleGoogleAuth = async () => {
    setError("");
    try {
      await signInWithPopup(auth, googleProvider);
      router.push("/");
    } catch (err: unknown) {
      setError((err as Error).message);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6" style={{ backgroundColor: 'var(--color-canvas-parchment)' }}>
      <div 
        className="w-full max-w-md p-[24px] flex flex-col items-center"
        style={{
          backgroundColor: 'var(--color-canvas)',
          borderRadius: '18px',
          border: '1px solid var(--color-hairline)'
        }}
      >
        <h1 className="display-lg mb-[24px] text-center w-full" style={{ color: 'var(--color-ink)' }}>
          {isLogin ? "Sign In" : "Create Account"}
        </h1>
        
        {error && (
          <div className="mb-[16px] w-full p-[12px] rounded-[8px] caption-text text-center" style={{ backgroundColor: '#fff0f0', color: '#cc0000' }}>
            {error}
          </div>
        )}

        <form onSubmit={handleEmailAuth} className="w-full flex flex-col gap-[16px]">
          <input 
            type="email" 
            placeholder="Email address"
            className="apple-search-input w-full"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input 
            type="password" 
            placeholder="Password"
            className="apple-search-input w-full"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          
          <div className="mt-[8px] flex flex-col gap-[12px]">
            <button type="submit" className="apple-button-primary w-full cursor-pointer">
              {isLogin ? "Sign In" : "Continue"}
            </button>
            
            <button 
              type="button"
              onClick={handleGoogleAuth}
              className="w-full cursor-pointer flex items-center justify-center gap-[8px]"
              style={{
                backgroundColor: 'var(--color-surface-pearl)',
                color: 'var(--color-ink)',
                borderRadius: '9999px',
                padding: '11px 22px',
                fontSize: '17px',
                lineHeight: '1.47',
                border: '1px solid var(--color-divider-soft)',
                transition: 'transform 0.15s ease-out'
              }}
              onMouseDown={(e) => e.currentTarget.style.transform = 'scale(0.95)'}
              onMouseUp={(e) => e.currentTarget.style.transform = 'scale(1)'}
              onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
            >
              <svg width="18" height="18" viewBox="0 0 24 24">
                <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
                <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
                <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
                <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
              </svg>
              Continue with Google
            </button>
          </div>
        </form>

        <div className="mt-[24px] w-full text-center body-text" style={{ color: 'var(--color-ink-muted-80)' }}>
          {isLogin ? "Don't have an account?" : "Already have an account?"}{' '}
          <button 
            onClick={() => setIsLogin(!isLogin)}
            className="cursor-pointer hover:underline"
            style={{ color: 'var(--color-primary)' }}
          >
            {isLogin ? "Create yours now." : "Sign In."}
          </button>
        </div>
      </div>
    </div>
  );
}
