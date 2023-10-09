import { Component } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { DomSanitizer } from '@angular/platform-browser';
import { environment } from 'src/environments/environment';

@Component({
  selector: 'app-topics',
  templateUrl: './topics.component.html',
  styleUrls: ['./topics.component.scss']
})
export class TopicsComponent {
  public count?: number;
  public term?: string;
  public start_search: boolean = false;
  readonly baseurl = environment.baseurl;

  constructor(
    public sanitizer: DomSanitizer,
  ) {
  }

}
