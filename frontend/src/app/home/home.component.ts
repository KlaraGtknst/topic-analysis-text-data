import { Component } from '@angular/core';
import { Document, HomeService } from './home.service';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent {
  public documents: Document[] = [];

  constructor(
    private homeService: HomeService,
  ) {
  }

  ngOnInit(): void {
    this.homeService.getdocs().subscribe(answer => {
      this.documents = answer;
    });
  }
}
