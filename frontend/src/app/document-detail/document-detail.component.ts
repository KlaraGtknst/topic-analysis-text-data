import { Component } from '@angular/core';
import { Document, DocumentService } from '../document.service';
import { ActivatedRoute } from '@angular/router';
import { switchMap } from 'rxjs';
import { environment } from 'src/environments/environment';
import { DomSanitizer } from '@angular/platform-browser';

@Component({
  selector: 'app-document-detail',
  templateUrl: './document-detail.component.html',
  styleUrls: ['./document-detail.component.scss']
})
export class DocumentDetailComponent {
  public doc?: Document;
  public queryType?: string;
  public similarDocs: Document[] = [];

  readonly queryTypes = ["doc2vec", "sim_docs_tfidf", "google_univ_sent_encoding", "huggingface_sent_transformer", "inferSent_AE", "pca_optics_cluster", "argmax_pca_cluster"];
  readonly baseurl = environment.baseurl;
  
  constructor(
    private documentService: DocumentService,
    public sanitizer: DomSanitizer,
    private route: ActivatedRoute,
  ) {
  }

  ngOnInit(): void {
    this.route.params.pipe( // put observable in pipeline
      switchMap(({ id }) => this.documentService.getdoc(id)), // cleans up old observables. ID property of current observable is further processed
    ).subscribe(answer => {
      this.doc = answer;
    });
  }
  
  findSimilar() {
    if (this.queryType && this.doc) {
      this.documentService.getSimilar(this.queryType, this.doc._id).subscribe(answer => {
        this.similarDocs = answer;
      });
    }
  }
}
