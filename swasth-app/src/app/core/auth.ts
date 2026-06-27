import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { UserService } from './user.service';

@Injectable({
  providedIn: 'root',
})
export class Auth {
  private apiUrl = 'https://swasth-healthcare-chatbot.onrender.com';

  constructor(
    private http:        HttpClient,
    private userService: UserService,
  ) {}

  login(data: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/login`, data);
  }

  register(data: any): Observable<any> {
    return this.http.post(`${this.apiUrl}/register`, data);
  }

  saveToken(token: string): void {
    localStorage.setItem('token', token);
  }

  getToken(): string | null {
    return localStorage.getItem('token');
  }

  /**
   * Decodes the JWT payload and checks whether the `exp` claim
   * has already passed. Returns true if expired (or unreadable).
   */
  isTokenExpired(): boolean {
    const token = this.getToken();
    if (!token) return true;

    try {
      // JWT = header.payload.signature — we only need the middle part.
      const payloadBase64 = token.split('.')[1];
      // atob decodes base64; replace handles URL-safe base64 chars.
      const payload = JSON.parse(atob(payloadBase64.replace(/-/g, '+').replace(/_/g, '/')));
      // `exp` is in seconds; Date.now() is in milliseconds.
      return payload.exp * 1000 < Date.now();
    } catch {
      // If we can't decode the token it is invalid — treat as expired.
      return true;
    }
  }

  /**
   * Returns true only when a token is present AND not yet expired.
   * This is the root-cause fix for "still logged in after 20 days".
   */
  isLoggedIn(): boolean {
    if (!this.getToken()) return false;

    if (this.isTokenExpired()) {
      // Auto-clear the stale token so guards redirect correctly.
      this.clearSession();
      return false;
    }

    return true;
  }

  /** Full logout: remove token + wipe in-memory profile cache. */
  logout(): void {
    this.clearSession();
  }

  /** Internal helper used by both logout() and isLoggedIn(). */
  private clearSession(): void {
    localStorage.removeItem('token');
    this.userService.clearProfile();
  }
}
