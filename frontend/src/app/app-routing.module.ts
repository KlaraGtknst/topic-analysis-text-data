import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import { HomeComponent } from './home/home.component';
import { DocumentDetailComponent } from './document-detail/document-detail.component';

const routes: Routes = [
  { path: ':id', component: DocumentDetailComponent},
  { path: '', component: HomeComponent},
  // TODO: wordcloud 
  // TODO: term frequency
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
