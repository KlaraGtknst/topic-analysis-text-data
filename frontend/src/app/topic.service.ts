import {Injectable} from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { environment } from 'src/environments/environment';
import { Observable } from 'rxjs';

export interface Topic {
    _id: string;
    _score?: number;
    terms: string[];
}

@Injectable({
  providedIn: 'root'
})
export class TopicService {


  constructor(private http: HttpClient) {}

  gettopics(term?: string, count?: number): Observable<Topic[]> {
    const params: any = {};
    if (term) {
        params.term = term;
    }
    if (count) {
        params.count = count;
    }
    return this.http.get<Topic[]>(environment.baseurl + 'topics', {
        params,
    });
  }


}