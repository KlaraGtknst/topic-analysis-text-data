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

  getdocs(searchText?: string): Observable<Document[]> {
    const params: any = {};
    if (searchText) {
        params.text = searchText;
    }
    return this.http.get<Document[]>(environment.baseurl + 'documents', {
        params,
    });
  }

}