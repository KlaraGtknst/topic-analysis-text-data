import { Component } from '@angular/core';
import { Document, DocumentService } from '../document.service';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})

export class HomeComponent {
  public documents: Document[] = [];
  searchText = '';
  
  constructor(
    private documentService: DocumentService,
  ) {
  }
  
  ngOnInit(): void {
    this.documentService.getdocs().subscribe(answer => {
      this.documents = answer;
    });
  }

  search() {
    this.documentService.getdocs(this.searchText).subscribe(answer => {
      this.documents = answer;
    });
  }
}
