import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class ChatService {

  private apiUrl = 'https://swasth-healthcare-chatbot.onrender.com/api';

  constructor(private http: HttpClient) {}

  private headers(): HttpHeaders {
    return new HttpHeaders({
      Authorization: `Bearer ${localStorage.getItem('token')}`
    });
  }

  // ── legacy ──────────────────────────────────────────────────────────────────
  sendSymptoms(symptoms: string[]): Observable<any> {
    return this.http.post(`${this.apiUrl}/chat`, { symptoms }, { headers: this.headers() });
  }

  // ── symptom search ───────────────────────────────────────────────────────────
  searchSymptoms(query: string): Observable<any> {
    const params = new HttpParams().set('q', query);
    return this.http.get(`${this.apiUrl}/symptoms/search`, { headers: this.headers(), params });
  }

  // ── assessment flow ──────────────────────────────────────────────────────────
  startAssessment(selectedSymptoms: string[]): Observable<any> {
    return this.http.post(`${this.apiUrl}/assess/start`,
      { selected_symptoms: selectedSymptoms },
      { headers: this.headers() }
    );
  }

  submitAnswer(payload: {
    session_id:         string;
    symptom:            string;
    answer:             string;
    confirmed_symptoms: string[];
    absent_symptoms:    string[];
    asked_symptoms:     string[];
    questions_asked:    number;
  }): Observable<any> {
    return this.http.post(`${this.apiUrl}/assess/answer`, payload, { headers: this.headers() });
  }

  getExplanation(payload: {
    session_id:  string;
    disease:     string;
    confidence:  number;
    symptoms:    string[];
  }): Observable<any> {
    return this.http.post(`${this.apiUrl}/assess/explain`, payload, { headers: this.headers() });
  }

  submitAssessmentFeedback(payload: {
    result_id?:         number;
    session_id:         string;
    feedback_type:      'like' | 'dislike';
    user_comment?:      string;
    predicted_disease:  string;
    symptoms:           string[];
    confidence:         number;
  }): Observable<any> {
    return this.http.post(`${this.apiUrl}/assess/feedback`, payload, { headers: this.headers() });
  }

  getRecommendations(payload: {
    disease:    string;
    confidence: number;
    symptoms:   string[];
    section:    'diet' | 'workout' | 'precautions';
  }): Observable<any> {
    return this.http.post(`${this.apiUrl}/assess/recommendations`, payload, { headers: this.headers() });
  }

  getHistory(): Observable<any> {
    return this.http.get(`${this.apiUrl}/history`, { headers: this.headers() });
  }
}
