import { Component } from '@angular/core';
import { ActivatedRoute } from '@angular/router';
import { DomSanitizer } from '@angular/platform-browser';
import { environment } from 'src/environments/environment';
import { Topic, TopicService } from '../topic.service';

@Component({
  selector: 'app-topics',
  templateUrl: './topics.component.html',
  styleUrls: ['./topics.component.scss'],
})
export class TopicsComponent {
  public count?: number;
  public term?: string;
  public start_search: boolean = false;
  public topics: Topic[] = [];
  readonly separatorSymbol = "_".repeat(200);
  readonly baseurl = environment.baseurl;

  constructor(
    public sanitizer: DomSanitizer,
    private topicService: TopicService,
  ) {
  }

  ngOnInit(): void {
    this.topicService.gettopics().subscribe(answer => {
      this.topics = answer;
      console.log(this.topics);
    });
  }

}
