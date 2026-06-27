import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable, BehaviorSubject, tap } from 'rxjs';

export interface UserProfile {
  id:       number;
  email:    string;
  fullName: string;
  age:      number | null;
  gender:   string | null;
  address:  string | null;
}

@Injectable({ providedIn: 'root' })
export class UserService {

  private apiUrl = 'https://swasth-healthcare-chatbot.onrender.com';

  // in-memory cache — cleared on logout
  private _profile$ = new BehaviorSubject<UserProfile | null>(null);
  readonly profile$ = this._profile$.asObservable();

  constructor(private http: HttpClient) {}

  private headers(): HttpHeaders {
    return new HttpHeaders({
      Authorization: `Bearer ${localStorage.getItem('token')}`
    });
  }

  /** Fetch profile from backend and cache it. Call once after login. */
  fetchProfile(): Observable<UserProfile> {
    return this.http
      .get<UserProfile>(`${this.apiUrl}/me`, { headers: this.headers() })
      .pipe(tap(profile => this._profile$.next(profile)));
  }

  /** Returns cached profile synchronously (null if not fetched yet). */
  getProfile(): UserProfile | null {
    return this._profile$.getValue();
  }

  /** Clear cached profile on logout. */
  clearProfile(): void {
    this._profile$.next(null);
  }
}
