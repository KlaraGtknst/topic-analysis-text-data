import {NgModule} from '@angular/core';
import {RouterModule, Routes} from '@angular/router';
import { HomeComponent } from './home/home.component';
import { DocumentDetailComponent } from './document-detail/document-detail.component';
import { TopicsComponent } from './topics/topics.component';

const routes: Routes = [
  { path: 'topics', component: TopicsComponent},
  { path: ':id', component: DocumentDetailComponent},
  { path: '', component: HomeComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
