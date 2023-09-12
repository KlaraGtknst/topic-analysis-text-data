import {Injectable} from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { environment } from 'src/environments/environment';
import { Observable } from 'rxjs';

export interface Document {
    _id: string;
    _score?: number;
    path: string;
    text: string;
}

@Injectable({
  providedIn: 'root'
})
export class HomeService {


  constructor(private http: HttpClient) {}

  getdocs(): Observable<Document[]> {
    return this.http.get<Document[]>(environment.baseurl + 'documents');
  }

}